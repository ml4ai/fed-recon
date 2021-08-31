import json
import os
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List, Union

import pandas as pd


class ExperimentLogger:
    def __init__(
        self,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None,
        create_output_dir: bool = True,
    ):
        if not create_output_dir:
            assert os.path.exists(output_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)
        if config is not None:
            config_path = os.path.join(output_dir, "config.json")
            assert not os.path.exists(
                config_path
            ), f"{config_path} exists. Make sure to use a new directory when logging an experiment."
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        self.output_dir = output_dir
        self.fn_headers: Dict[str, Tuple[str]] = {}  # Map from filename to csv headers
        self.fn_cols: Dict[str, Dict[str, List[Union[float, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )  # Map from filename to column name to list of column values

    def record_result(
        self, filename: str, header: Tuple[str], row: List[Union[float, str]]
    ):
        filename = os.path.join(self.output_dir, filename)
        assert len(row) == len(header)
        if filename not in self.fn_headers:
            self.fn_headers[filename] = header
        else:
            assert self.fn_headers[filename] == header

        for col, val in zip(header, row):
            self.fn_cols[filename][col].append(val)

    def serialize_results(self):
        for filename, data in self.fn_cols.items():
            df = pd.DataFrame(data)
            df.to_csv(filename, header=True, index=False)
            print(f"wrote results to {filename}")
