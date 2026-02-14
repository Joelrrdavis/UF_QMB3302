# Week 3 simple csv
# alt_csv_path = Path("data") / "sample_alt.csv"

if not alt_csv_path.exists():
    df_alt_make = pd.DataFrame(
        {
            "student": ["Taylor", "Avery", "Quinn"],
            "exam_score": [88, 91, 79],
        }
    )
    df_alt_make.to_csv(alt_csv_path, index=False)



## ---------------------------------------------
# Student Test scores file generator
## ---------------------------------------------

# We will load data from "data/example1.csv".
# This path is RELATIVE to the current working directory (CWD).
test_scores = Path("data") / "student_test_results.csv"
print(test_scores)

if not test_scores.exists():

    # Create the parent folder ("data/") if needed.
    test_scores.parent.mkdir(parents=True, exist_ok=True)

    # This dataset is intentionally messy:
    # - Missing values
    # - A column name with spaces
    # - A column we will later drop
    df_make = pd.DataFrame(
        {
            "Student Name": ["Alex", "Jordan", "Casey", "Riley", "Morgan"],
            "score": [90, 85, None, 72, 93],
            "major": ["Finance", "Marketing", "IS", None, "IS"],
            "unused_column": ["x", "x", "x", "x", "x"],
        }
    )

    df_make.to_csv(test_scores, index=False)

    print("CSV not found — created a sample dataset.")

else:
    print("CSV already exists — using the existing file.")
