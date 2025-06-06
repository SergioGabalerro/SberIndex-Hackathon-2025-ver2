import logging
import pandas as pd

from ..data_loader import DataLoader


class KnowledgeSearchAgent:
    def __init__(self, data_files: dict):
        self.role = "Поисковик"
        self.data_files = data_files
        self.data = DataLoader.load_data(data_files)
        logging.info(f"[{self.role}] Инициализирован.")

    def combine_dataframes(self, datasets: list) -> pd.DataFrame:
        """Объединяет указанные датафреймы по ключу territory_id."""
        dfs_to_merge = []
        for ds in datasets:
            if ds in self.data:
                dfs_to_merge.append(self.data[ds].copy())
        if "mo_ref" in self.data and "mo_ref" not in datasets:
            dfs_to_merge.append(self.data["mo_ref"].copy())

        if not dfs_to_merge:
            return pd.DataFrame()

        combined = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            if "territory_id" in combined.columns and "territory_id" in df.columns:
                combined = combined.merge(df, on="territory_id", how="inner")
            else:
                combined = pd.concat([combined, df], axis=1)
        return combined