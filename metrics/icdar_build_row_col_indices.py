from operator import itemgetter


class BuilderRowColIndices():

    def build_row_col_indices(self, cells_dict):
        """
        Having a list of cell postitions build row and column indices that
        correspond to every cell for a batch of tables
        Args:
            cells_dict: {table_name: {cells_list: a list of cells [xyxy]},..}
        Returns:
            cells_dict: {table_name:
            {"cells_col_row": [(start_col, start_row, end_col, end_row),..]},.}
            if a cell is not spanning then end_col, end_row = 0
        """
        for table_name in cells_dict.keys():
            cells_list = cells_dict[table_name]["cells_list"]

            # Columns
            col_ranges = self._build_ranges(cells_list, "column")
            col_list = self._build_indices(cells_list, col_ranges, "column")

            # Rows
            row_ranges = self._build_ranges(cells_list, "row")
            row_list = self._build_indices(cells_list, row_ranges, "row")
            col_row_list = []
            for col, row in zip(col_list, row_list):
                if len(col) == 1:
                    start_col = col[0]
                    end_col = -1
                else:
                    start_col = min(col)
                    end_col = max(col)
                if len(row) == 1:
                    start_row = row[0]
                    end_row = -1
                else:
                    start_row = min(row)
                    end_row = max(row)
                col_row_list.append([start_col, start_row, end_col, end_row])
            cells_dict[table_name]["cells_col_row"] = col_row_list

        return cells_dict

    def _build_ranges(self, cells_list, dataclass):
        """
        Build a list of pixel positions where every column/row starts and ends

        Algorithm for finding column ranges:
        1. Sort cells by right border of cells from leftmost to rightmost
        2. The cell with the leftmost right border define column
        3. Throw away all cells that intersect with the cell from step 2 for
        more than 20%
        4. Repeat steps 1 and 2 to define next column

        Algorithm for finding row ranges is analogical.
        """
        cells_list_copy = cells_list.copy()
        ranges = []
        while len(cells_list_copy) > 0:
            if dataclass == "column":
                cells_list_copy.sort(key=itemgetter(2))
                left, top, right, bottom = cells_list_copy[0]
                ranges.append((left, right))
            elif dataclass == "row":
                cells_list_copy.sort(key=itemgetter(3))
                left, top, right, bottom = cells_list_copy[0]
                ranges.append((top, bottom))
            idx_to_del = []
            for idx, cell in enumerate(cells_list_copy):
                x1, y1, x2, y2 = cell
                if dataclass == "column":
                    intersect = self._intersection_lines(left, right, x1, x2)
                elif dataclass == "row":
                    intersect = self._intersection_lines(top, bottom, y1, y2)
                if intersect:
                    idx_to_del.append(idx)
            cells_list_copy = self._del_indices_from_list(
                idx_to_del, cells_list_copy
            )
        return ranges

    def _build_indices(self, cells_list, ranges, dataclass):
        """
        Build a list of column/row numbers that correspond to the
        given columns/rows ranges
        """
        idx_list = [-1] * len(cells_list)
        for idx, cell in enumerate(cells_list):
            x1, y1, x2, y2 = cell
            for idx_range, range in enumerate(ranges):
                if dataclass == "column":
                    left, right = range
                    intersect = self._intersection_lines(left, right, x1, x2)
                elif dataclass == "row":
                    top, bottom = range
                    intersect = self._intersection_lines(top, bottom, y1, y2)
                if intersect:
                    if idx_list[idx] != -1:
                        # Spanning cell
                        idx_list[idx].append(idx_range)
                    else:
                        idx_list[idx] = [idx_range]
        return idx_list

    def _intersection_lines(self, start, end, a, b):
        """
        Check if (a,b) overlaping with (start,end) for more than 20%
        """
        if a >= end or b <= start:
            return False
        return True

    def _del_indices_from_list(self, idx_to_delete, list_):
        for index in sorted(idx_to_delete, reverse=True):
            del list_[index]
        return list_
