from operator import itemgetter


class AdjacencyRelationsGenerator():

    def _max_idx(self, cells_col_row_list, dataclass):
        """
        Computes the maximum column/row index in the table
        """
        if dataclass == "column":
            idx1 = 0  # start_col
            idx2 = 2  # end_col
        elif dataclass == "row":
            idx1 = 1  # start_row
            idx2 = 3  # end_row
        start_max = max(cells_col_row_list, key=itemgetter(idx1))  # start_col
        end_max = max(cells_col_row_list, key=itemgetter(idx2))  # end_col

        return max(start_max, end_max)

    def _build_set(self, start_, end_):
        """
        Given start_ and end_ indices build a set of all indices between them
        (incl. them)
        """
        if end_ == -1:
            return set([start_])
        return set(range(start_, end_ + 1))

    def _build_adj(self, start_, end_, max_):
        """
        For columns: finds starting index of adjacent column to the right of
            the given column (could be spanning for several columns)
        For rows: finds starting index of adjacent row to the down of the given
            row (could be spanning for several rows)

        Args:
            start_: starting index of the column/row
            end_: ending index of the column/row
            max_: maximum possible index of the column/row in the given table
        Returns:
            an index of adjacent column/row, or -1 if no adjacent column/row
        Examples:
            build_adj(3, -1, 5)=> 4
            build_adj(3, 4, 5)=> 5
            build_adj(3, 5, 5)=> -1
        """
        adj_right_down = -1
        if end_ == -1 and start_ != max_:
            adj_right_down = start_ + 1
        if end_ != -1 and end_ != max_:
            adj_right_down = end_ + 1
        return adj_right_down


class CtdarAdjacencyRelationsGenerator(AdjacencyRelationsGenerator):
    def generate_adj_rel(self, cells_dict):
        """
        Generate the adjacency relationships list.
        row and col indices starts from 0.
        It is enough to only describe adjacency relationships for every cell in
        the direction to the right and down.
        Args:
            cells_dict: {'table_name':
            {"cells_col_row": [[start_col, start_row, end_col, end_row],..],
            "cells_list": [[0,217,89,262],..]}
            }
        Returns:
            an adjacency relationship list for every table
            Example: [[0,217,89,262],[0,382,83,425],"down",1],...
        """
        adj_rel_dict = {}
        for table_name, cells in cells_dict.items():
            adj_rel_list = []
            cells_col_row_list = cells["cells_col_row"]
            cells_list = cells["cells_list"]

            # Find the index of the rightmost column and bottommost row
            col_max = self._max_idx(cells_col_row_list, "column")
            row_max = self._max_idx(cells_col_row_list, "row")

            for idx, cell1 in enumerate(cells_list):
                start_col, start_row, end_col, end_row = \
                    cells_col_row_list[idx]

                # Build a set of column and row indices that the cell1 covers
                col = self._build_set(start_col, end_col)
                row = self._build_set(start_row, end_row)

                # Build a list of the adjacent columns and rows
                adj_col_right = self._build_adj(start_col, end_col, col_max)
                adj_row_down = self._build_adj(start_row, end_row, row_max)

                # Build a list of zipped cells_col_row_list and cells_list
                # excluding cell1
                cells_col_row_list_copy = cells_col_row_list.copy()
                del cells_col_row_list_copy[idx]
                cells_list_copy = cells_list.copy()
                del cells_list_copy[idx]
                col_row_cells = list(
                    zip(cells_col_row_list_copy, cells_list_copy)
                )

                # Build adjacency relationships of the given cell with the
                # adjacent cells in the direction of down
                adj_rel_list_add = self._find_adj_cells(
                    adj_row_down, col_row_cells, "row", cell1, row, col)
                adj_rel_list.extend(adj_rel_list_add)

                # Build adjacency relationships of the given cell with the
                # adjacent cells in the direction of right
                adj_rel_list_add = self._find_adj_cells(
                    adj_col_right, col_row_cells, "column", cell1, row, col)
                adj_rel_list.extend(adj_rel_list_add)

            adj_rel_dict[table_name] = adj_rel_list
        return adj_rel_dict

    def _find_adj_cells(
        self, adj, col_row_cells, dataclass, cell1, row1, col1
    ):
        """
        Args:
            adj: index of the adjacent column/row
            col_row_cells: a list of all cells in the table with their
                (start_col, start_row, end_col, end_row)
            dataclass: column/row
            cell1: the given cell
            row1: the set of rows covered by the given cell1
            col1: the set of columns covered by the given cell1

        Returns:
            a list of adjacency relationships for a given cell to the
            right/down
        """
        adj_rel_list = []
        if adj == -1:
            return adj_rel_list
        empty_cells = 0
        found = False
        if dataclass == "column":
            # Sort by start_col2
            col_row_cells = sorted(col_row_cells, key=lambda x: x[0][0])
            for col_row, cell2 in col_row_cells:
                start_col2, start_row2, end_col2, end_row2 = col_row
                # If we have iterated over all cells where start_col2 == adj
                # and haven't found adjacent 'right' cell, then it means that
                # adjacent cell was empty
                if start_col2 > adj and not found:
                    adj += 1
                    empty_cells += 1
                if start_col2 == adj:
                    row2 = self._build_set(start_row2, end_row2)
                    row_intersect = row1.intersection(row2)
                    if not row_intersect:
                        continue
                    if adj_rel_list:
                        adj_rel_list.append(
                            [cell1, cell2, "right", empty_cells]
                        )
                    else:
                        adj_rel_list = [[cell1, cell2, "right", empty_cells]]
                    found = True
        elif dataclass == "row":
            # Sort by start_row2
            col_row_cells = sorted(col_row_cells, key=lambda x: x[0][1])
            for col_row, cell2 in col_row_cells:
                start_col2, start_row2, end_col2, end_row2 = col_row
                # If we iterated over all cells where start_row2 == adj
                # and haven't found adjacent 'down' cell, then it means that
                # adjacent cell was empty
                if start_row2 > adj and not found:
                    adj += 1
                    empty_cells += 1
                if start_row2 == adj:
                    col2 = self._build_set(start_col2, end_col2)
                    col_intersect = col1.intersection(col2)
                    if not col_intersect:
                        continue
                    if adj_rel_list:
                        adj_rel_list.append(
                            [cell1, cell2, "down", empty_cells]
                        )
                    else:
                        adj_rel_list = [[cell1, cell2, "down", empty_cells]]
                    found = True
        return adj_rel_list


class IcdarAdjacencyRelationsGenerator(AdjacencyRelationsGenerator):

    def generate_adj_rel(self, cells_dict):
        """
        Generate the adjacency relationships list.
        row and col indices starts from 0.
        It is enough to only describe adjacency relationships for every cell in
        the direction to the right and down.
        Args:
            gt_tables_dict: {'table_name':
            {"cells_col_row": [[start_col, start_row, end_col, end_row],..],
            "contents_list": ["Text1","Text2",..]}
            }
        Returns:
            an adjacency relationship list for every table
            Example: ["Text1","Text2",down",1],...
        """
        table_strs = {}
        for table_name, cells in cells_dict.items():
            print("table_name: ", table_name)
            contents = cells["contents_list"]
            cells_col_row_list = cells["cells_col_row"]
            table_str = []

            # Find the index of the rightmost column and lowest row
            col_max = self._max_idx(cells_col_row_list, "column")
            row_max = self._max_idx(cells_col_row_list, "row")

            # Content stripped of all spaces and special characters
            for idx, content in enumerate(contents):
                contents[idx] = ''.join(
                    char for char in content if char.isalnum()
                ).lower()

            # Remove empty cells from predicted contents
            # (in ground-truth no empty cells)
            contents, cells_col_row_list = self._remove_empty_cells(
                contents, cells_col_row_list
            )

            for idx, content1 in enumerate(contents):
                start_col, start_row, end_col, end_row = \
                    cells_col_row_list[idx]

                # Build a set of column and row indices that the cell covers
                col = self._build_set(start_col, end_col)
                row = self._build_set(start_row, end_row)

                # Build a list of the adjacent columns and rows
                adj_col_right = self._build_adj(start_col, end_col, col_max)
                adj_row_down = self._build_adj(start_row, end_row, row_max)

                # Build a list of zipped cells_col_row_list and contents
                # excluding cell1
                cells_col_row_list_copy = cells_col_row_list.copy()
                del cells_col_row_list_copy[idx]
                contents_copy = contents.copy()
                del contents_copy[idx]
                col_row_contents = list(
                    zip(cells_col_row_list_copy, contents_copy)
                )

                # Build adjacency relationships of the given cell with the
                # adjacent cells in the direction of down
                table_str_add = self._find_adj_cells(
                    adj_row_down, col_row_contents, "row", content1, row, col)
                table_str.extend(table_str_add)

                # Build adjacency relationships of the given cell with the
                # adjacent cells in the direction of right
                table_str_add = self._find_adj_cells(adj_col_right,\
                    col_row_contents, "column", content1, row, col)
                table_str.extend(table_str_add)

            table_strs[table_name] = table_str
        return table_strs

    def _find_adj_cells(
        self, adj, col_row_contents, dataclass, content1, row1, col1
    ):
        """
        Args:
            adj: index of the adjacent column/row
            col_row_cells: a list of all cells in the table with their
                (start_col, start_row, end_col, end_row)
            dataclass: column/row
            cell1: the given cell
            row1: the set of rows covered by the given cell1
            col1: the set of columns covered by the given cell1

        Returns:
            a list of adjacency relationships for a given cell to the
            right/down
        """
        table_str = []
        if adj == -1:
            return table_str
        empty_cells = 0
        found = False
        if dataclass == "column":
            # Sort by start_col2
            col_row_contents = sorted(col_row_contents, key=lambda x: x[0][0])
            for col_row, content2 in col_row_contents:
                start_col2, start_row2, end_col2, end_row2 = col_row
                # If we iterated over all cells where start_col2 == adj
                # and haven't found adjacent right cell, then it means that
                # adjacent cell was empty
                if start_col2 > adj and not found:
                    adj += 1
                    empty_cells += 1
                if start_col2 == adj:
                    row2 = self._build_set(start_row2, end_row2)
                    row_intersect = row1.intersection(row2)
                    if row_intersect:
                        if table_str:
                            table_str.append(
                                (content1, content2, "right", empty_cells)
                            )
                        else:
                            table_str = [
                                (content1, content2, "right", empty_cells)
                            ]
                        found = True
        elif dataclass == "row":
            # Sort by start_row2
            col_row_contents = sorted(col_row_contents, key=lambda x: x[0][1])
            for col_row, content2 in col_row_contents:
                start_col2, start_row2, end_col2, end_row2 = col_row
                # If we iterated over all cells where start_row2 == adj
                # and haven't found adjacent right cell, then it means that
                # adjacent cell was empty
                if start_row2 > adj and not found:
                    adj += 1
                    empty_cells += 1
                if start_row2 == adj:
                    col2 = self._build_set(start_col2, end_col2)
                    col_intersect = col1.intersection(col2)
                    if col_intersect:
                        if table_str:
                            table_str.append(
                                (content1, content2, "down", empty_cells)
                            )
                        else:
                            table_str = [
                                (content1, content2, "down", empty_cells)
                            ]
                        found = True
        return table_str

    def _remove_empty_cells(self, contents, cells_col_row_list):
        """
        Remove simultaneously elements from contents and cells_col_row_list,
        if the content is empty
        """
        contents_copy = contents.copy()
        for idx, content in reversed(list(enumerate(contents))):
            if not content:
                del contents_copy[idx]
                del cells_col_row_list[idx]
        return contents_copy, cells_col_row_list
