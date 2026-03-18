from dende_statistics import Statistics
from typing import Dict, List, Set, Any

class MissingValueProcessor:
    
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        if columns is not None:
            return list(columns)
        return list(self.dataset.keys())

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_columns = self._get_target_columns(columns)
        all_columns = list(self.dataset.keys())
        result_dataset = {}
        for col in all_columns:
            result_dataset[col] = []
        
        if not all_columns:
            return result_dataset

        row_count = len(self.dataset[all_columns[0]])

        invalid_rows = set()

        for row_index in range(row_count):
            for col in target_columns:
                if self.dataset[col][row_index] is None:
                    invalid_rows.add(row_index)

        for row_index in invalid_rows:
            for col_all in all_columns:
                result_dataset[col_all].append(self.dataset[col_all][row_index])
                
        return result_dataset

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_columns = self._get_target_columns(columns)
        all_columns = list(self.dataset.keys())
        result_dataset = {}
        for col in all_columns:
            result_dataset[col] = []
        
        if not all_columns:
            return result_dataset

        row_count = len(self.dataset[all_columns[0]])

        invalid_rows = set()

        for row_index in range(row_count):
            for col in target_columns:
                if self.dataset[col][row_index] is None:
                    invalid_rows.add(row_index)

        for row_index in invalid_rows:
            for col_all in all_columns:
                result_dataset[col_all].append(self.dataset[col_all][row_index])
                
        return result_dataset

    def fillna(self, columns: Set[str] = None, value: Any = 0) -> Dict[str, List[Any]]:
        target_columns = self._get_target_columns(columns)

        if not self.dataset:
            return self.dataset
            
        all_columns = list(self.dataset.keys())
        row_count = len(self.dataset[all_columns[0]])

        for col in target_columns:
            for row_index in range(row_count):
                if self.dataset[col][row_index] is None:
                    self.dataset[col][row_index] = value
                    
        return self.dataset

    def dropna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_columns = self._get_target_columns(columns)
        all_columns = list(self.dataset.keys())
        
        if not all_columns:
            return self.dataset
            
        row_count = len(self.dataset[all_columns[0]])
        valid_indices = []

        for row_index in range(row_count):
            is_valid = True
            for col in target_columns:
                if self.dataset[col][row_index] is None:
                    is_valid = False
            if is_valid:
                valid_indices.append(row_index)
        
        new_dataset = {}
        for col in all_columns:
            new_dataset[col] = []
        
        for row_index in valid_indices:
            for col in all_columns:
                new_dataset[col].append(self.dataset[col][row_index])

        for col in all_columns:
            self.dataset[col] = new_dataset[col]

        return self.dataset


class Scaler:

    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        if columns:
            return list(columns)
        return list(self.dataset.keys())

    def minMax_scaler(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_columns = self._get_target_columns(columns)
        
        if not self.dataset:
            return self.dataset
            
        all_columns = list(self.dataset.keys())
        row_count = len(self.dataset[all_columns[0]])

        for col in target_columns:
            
            valid_values = []
            for val in self.dataset[col]:
                if val is not None:
                    valid_values.append(val)            
            
            if not valid_values:
                continue
                
            min_val = min(valid_values)
            max_val = max(valid_values)
            range_val = max_val - min_val

            if range_val == 0:
                for row_index in range(row_count):
                    if self.dataset[col][row_index] is not None:
                        self.dataset[col][row_index] = 0.0
            else:
                for row_index in range(row_count):
                    if self.dataset[col][row_index] is not None:
                        current_val = self.dataset[col][row_index]
                        self.dataset[col][row_index] = (current_val - min_val) / range_val

        return self.dataset

    def standard_scaler(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_columns = self._get_target_columns(columns)

        if not self.dataset:
            return self.dataset

        all_columns = list(self.dataset.keys())
        row_count = len(self.dataset[all_columns[0]])

        stats = Statistics(self.dataset)

        for col in target_columns:

            valid_values = []
            for val in self.dataset[col]:
                if val is not None:
                    valid_values.append(val)

            if not valid_values:
                continue

            mean_val = stats.mean(col)
            stdev_val = stats.stdev(col)

            if stdev_val is None:
                continue

            if stdev_val != 0:
                for row_index in range(row_count):
                    if self.dataset[col][row_index] is not None:
                        val = self.dataset[col][row_index]
                        self.dataset[col][row_index] = (val - mean_val) / stdev_val
            else:
                for row_index in range(row_count):
                    if self.dataset[col][row_index] is not None:
                        self.dataset[col][row_index] = 0.0

        return self.dataset

class Encoder:

    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def label_encode(self, columns: Set[str]) -> Dict[str, List[Any]]:
        if not self.dataset:
            return self.dataset

        for col in columns:

            unique_values = []
            for val in self.dataset[col]:
                if val not in unique_values:
                    unique_values.append(val)

            unique_values.sort(key=str)

            mapping = {}
            index = 0
            for val in unique_values:
                mapping[val] = index
                index += 1

            for i in range(len(self.dataset[col])):
                current_val = self.dataset[col][i]
                self.dataset[col][i] = mapping[current_val]

        return self.dataset
    
    def one_hot_encode(self, columns: Set[str]) -> Dict[str, List[Any]]:
        if not self.dataset:
            return self.dataset

        for col in columns:

            unique_values = []
            for val in self.dataset[col]:
                if val not in unique_values:
                    unique_values.append(val)

            for val in unique_values:
                new_col_name = col + "_" + str(val)
                self.dataset[new_col_name] = []

            for i in range(len(self.dataset[col])):
                current_val = self.dataset[col][i]

                for val in unique_values:
                    new_col_name = col + "_" + str(val)

                    if current_val == val:
                        self.dataset[new_col_name].append(1)
                    else:
                        self.dataset[new_col_name].append(0)

            del self.dataset[col]

        return self.dataset

class Preprocessing:

    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        self._validate_dataset_shape()
        
        self.statistics = Statistics(self.dataset)
        self.missing_values = MissingValueProcessor(self.dataset)
        self.scaler = Scaler(self.dataset)
        self.encoder = Encoder(self.dataset)

    def _validate_dataset_shape(self):
        columns_lists = list(self.dataset.values())
        
        if not columns_lists:
            return

        reference_length = len(columns_lists[0])

        for column_data in columns_lists:
            if len(column_data) != reference_length:
                raise ValueError("Inconsistent list lengths")

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        return self.missing_values.isna(columns)

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        return self.missing_values.notna(columns)

    def fillna(self, columns: Set[str] = None, value: Any = 0) -> Dict[str, List[Any]]:
        return self.missing_values.fillna(columns, value)

    def dropna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        return self.missing_values.dropna(columns)

    def scale(self, columns: Set[str] = None, method: str = 'minMax') -> Dict[str, List[Any]]:
        if method == 'minMax':
            return self.scaler.minMax_scaler(columns)
        elif method == 'standard':
            return self.scaler.standard_scaler(columns)
        else:
            raise ValueError("Method not supported")

    def encode(self, columns: Set[str], method: str = 'label') -> Dict[str, List[Any]]:
        if method == 'label':
            return self.encoder.label_encode(columns)
        elif method == 'oneHot':
            return self.encoder.one_hot_encode(columns)
        else:
            raise ValueError("Method not supported")
