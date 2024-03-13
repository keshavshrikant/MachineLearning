import json
import pandas as pd

class Dolly:
    def __init__(self, instructions_data_path):
        self.instructions_data_path = instructions_data_path

    def read_databricks_dolly_data(self):
        features = []
        
        with open(self.instructions_data_path, 'r') as file:
            for line in file:
                features.append(json.loads(line))
                
        return pd.DataFrame.from_records(features)
                                
                
# if __name__ == "__main__":
#     instr = Intructions("databricks-dolly-15k.jsonl")
#     ret_df = instr.read_databricks_dolly_data()
#     print(ret_df.head())