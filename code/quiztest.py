import pandas as pd

data = pd.DataFrame({
 "group": ["A", "A", "B","A", "A", "A","A", "A", "A",
         "A", "B", "B", "A","B", "B", "A","A", "B", "B", "B"],
 "value": [11, 11, 12, 12, 12, 12, 12, 13, 13, 13,
          12, 12, 12, 12, 12, 13, 13, 13, 13, 13]
})
data.groupby("group")["value"].describe()