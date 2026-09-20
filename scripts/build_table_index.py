#!/usr/bin/env python3
"""M1-2: 389 份 Excel(.xls/.xlsx)→ 行级取数记录索引。

对每张工作表:
1. 读取值网格并回填合并单元格(取区域左上值);
2. 识别表头行:含期间模式(YYYY年M月/季度/年度)或口径词(本月/本年累计/当期/同比);
3. 列标签 = 各表头行非空单元格拼接(如"本年累计/截至当期");
4. 数据行:首列指标名(叠加上方分类行构建指标路径),逐列产出记录:
   (doc_id, sheet, row, col, indicator_path, column_label, value, unit)

用法: python3 scripts/build_table_index.py
"""
import json
import os
import re
import sys

import openpyxl
import xlrd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_INDEX = os.path.join(ROOT, "data", "processed", "doc_index.json")
OUT = os.path.join(ROOT, "data", "processed", "table_records.jsonl")

PERIOD_RE = re.compile(
    r"(20\d{2})\s*年(?:\s*(\d{1,2})\s*月|([一二三四]|\d{1,2})\s*季度|年度|全年)?")
CALIBER_RE = re.compile(
    r"本月|本年累计|累计|当期|本期|同比|同期|比年初|余额|发生额|上年末|年初|"
    r"[一二三四1-4]\s*季度|[一二三四]季度末|[1-9]\s*月(?:末|度)?")
NUM_RE = re.compile(r"^-?\d+(?:,\d{3})*(?:\.\d+)?$")


def norm_cell(v) -> str:
    return re.sub(r"\s+", "", str(v)) if v is not None else ""


def to_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("，", "")
    if NUM_RE.match(s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


def grid_xlsx(path: str):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
    for ws in wb.worksheets:
        merged = [list(map(int, [r.min_row - 1, r.max_row - 1,
                                 r.min_col - 1, r.max_col - 1]))
                  for r in ws.merged_cells.ranges]
        rows = [[c.value for c in row] for row in ws.iter_rows()]
        yield ws.title, rows, merged


def grid_xls(path: str):
    wb = xlrd.open_workbook(path, formatting_info=False)
    for ws in wb.sheets():
        rows = [[ws.cell_value(r, c) if c < ws.ncols else None
                 for c in range(ws.ncols)] for r in range(ws.nrows)]
        yield ws.name, rows, list(ws.merged_cells)


def fill_merged(rows, merged):
    for (r1, r2, c1, c2) in merged:  # 0-based, [r1,r2) x [c1,c2)
        v = rows[r1][c1] if r1 < len(rows) and c1 < len(rows[r1]) else None
        for r in range(r1, min(r2, len(rows))):
            for c in range(c1, min(c2, len(rows[r]) if r < len(rows) else 0) or c2 + 1):
                if c < len(rows[r]):
                    rows[r][c] = v
    return rows


def detect_header(rows):
    """返回 (header_row_idxs, unit_row_idx)。

    主规则:含 ≥1 个"短"期间/口径单元格(≤16字,排除标题行)且本行无数值;
    回退规则:无期间/口径词的表(如"地区×险种"型),取首个数值行之上、
    最近的"纯文本且 ≥2 个非空格"的行作表头。
    单位行单独全表扫描。"""
    def has_num(row):
        return any(to_number(v) is not None for v in row)

    header_rows = []
    for r, row in enumerate(rows):
        cells = [norm_cell(v) for v in row]
        n_short = sum(1 for c in cells
                      if c and len(c) <= 16 and (PERIOD_RE.search(c) or CALIBER_RE.search(c)))
        if n_short >= 1 and not has_num(row):
            header_rows.append(r)
        elif header_rows and has_num(row):
            break
        elif header_rows and r > header_rows[-1] + 3:
            break

    if not header_rows:  # 回退:首个数值行上方的最近文本表头
        for r, row in enumerate(rows):
            if has_num(row):
                for up in range(r - 1, max(-1, r - 4), -1):
                    cells = [norm_cell(v) for v in rows[up]]
                    n_nonempty = sum(1 for c in cells if c)
                    if n_nonempty >= 2 and not has_num(rows[up]):
                        header_rows = [up]
                        break
                break

    unit_row = None
    for r, row in enumerate(rows):
        if re.search(r"单位\s*[:：]", "".join(norm_cell(v) for v in row)):
            unit_row = r
            break
    return header_rows, unit_row


def column_labels(rows, header_rows, ncols):
    labels = [""] * ncols
    for r in header_rows:
        if r >= len(rows):
            continue
        for c, v in enumerate(rows[r][:ncols]):
            t = norm_cell(v)
            if t and t not in ("", "—", "-"):
                labels[c] = (labels[c] + "/" + t) if labels[c] else t
    return labels


def emit_sheet(doc_id, sheet_name, rows, merged, fp):
    if not rows:
        return 0
    ncols = max(len(r) for r in rows)
    rows = fill_merged([list(r) + [None] * (ncols - len(r)) for r in rows], merged)
    header_rows, unit_row = detect_header(rows)
    if not header_rows:
        return 0
    unit = ""
    if unit_row is not None and unit_row < len(rows):
        m = re.search(r"单位\s*[:：]\s*(.+)", "".join(norm_cell(v) for v in rows[unit_row]))
        unit = m.group(1) if m else ""

    labels = column_labels(rows, header_rows, ncols)
    first_data = max(header_rows) + 1
    # 指标列:前几列中"文本为主、数值稀少"且非空率最高的列(通常第1或第2列)
    ind_col = 0
    best = 0.0
    data_n = max(1, len(rows) - first_data)
    for c in range(min(4, ncols)):
        col_cells = [rows[r][c] for r in range(first_data, len(rows)) if c < len(rows[r])]
        nonempty = [norm_cell(v) for v in col_cells if norm_cell(v)]
        if not nonempty:
            continue
        numeric_share = sum(1 for v in col_cells if to_number(v) is not None) / data_n
        text_share = len(nonempty) / data_n
        score = text_share * (1 - numeric_share)
        if score > best:
            best, ind_col = score, c
    if best == 0:  # 全数值表,退回首列
        ind_col = 0

    category = ""   # 分类行:指标列有文本但整行无数值
    top_level = ""  # 家族根:最后一个非"其中:"/非缩进的顶级指标
    n = 0
    for r in range(first_data, len(rows)):
        row = rows[r]
        ind = norm_cell(row[ind_col]) if ind_col < len(row) else ""
        if not ind:
            continue
        # 单位行可能在表中间再次出现(亿元区块/万件区块),出现即更新当前单位
        m_unit = re.search(r"单位\s*[:：]\s*(.+)", "".join(norm_cell(v) for v in row))
        numeric_vals = [to_number(v) for v in row]
        has_number = any(v is not None for v in numeric_vals)
        if m_unit and not has_number:
            unit = m_unit.group(1)
            category = ""
            continue
        if not has_number:
            category = ind
            continue
        # 家族归属:"其中:"前缀或缩进行 = 上一顶级指标的子项
        is_child = ind.startswith("其中") or ind != ind.lstrip()
        base_ind = ind.lstrip()
        if is_child and top_level:
            family_root = top_level
        else:
            family_root = base_ind
            top_level = base_ind
        ind_path = f"{category}/{base_ind}" if category and category != base_ind else base_ind
        for c in range(ncols):
            if c == ind_col:
                continue
            v = to_number(row[c])
            if v is None or not labels[c]:
                continue
            fp.write(json.dumps({
                "doc_id": doc_id, "sheet": sheet_name, "row": r + 1, "col": c + 1,
                "indicator": base_ind, "indicator_path": ind_path,
                "family_root": family_root,
                "column_label": labels[c], "value": v, "unit": unit,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    docs = json.load(open(DOC_INDEX, encoding="utf-8"))["docs"]
    sheet_docs = [d for d in docs if d["type"] in ("xls", "xlsx")]
    total_records, n_sheets, failed = 0, 0, []
    with open(OUT, "w", encoding="utf-8") as fp:
        for d in sheet_docs:
            did, path = d["doc_id"], os.path.join(ROOT, d["canonical_path"])
            try:
                grids = grid_xlsx(path) if d["type"] == "xlsx" else grid_xls(path)
                for name, rows, merged in grids:
                    total_records += emit_sheet(did, name, rows, merged, fp)
                    n_sheets += 1
            except Exception as e:  # noqa: BLE001
                failed.append({"doc_id": did, "error": f"{type(e).__name__}: {e}"})

    print(f"工作簿 {len(sheet_docs)} 份 / 工作表 {n_sheets} 张 → {total_records} 条取数记录")
    print(f"输出: {os.path.relpath(OUT, ROOT)}")
    if failed:
        print(f"失败 {len(failed)}:")
        for f in failed[:10]:
            print("  ", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
