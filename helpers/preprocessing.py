import pandas as pd

precautions_df = pd.read_csv("data/symptom_precaution.csv")
symptom_description_df = pd.read_csv("data/symptom_Description.csv")
severity_df = pd.read_csv("data/Symptom-severity.csv")
main_df = pd.read_csv("data/dataset.csv")

# strip whitespace
symptom_columns = [col for col in main_df.columns if "Symptom" in col]
for col in symptom_columns:
    main_df[col] = main_df[col].str.strip()

# create severity dict to tally severities to show to end-user
severity_dict = severity_df.set_index("Symptom")["weight"].to_dict()


def calculate_severity(row):
    severity_sum = 0
    for symptom in row[1:]:
        if symptom in severity_dict:
            severity_sum += severity_dict[symptom]
    return severity_sum


# add severity_tally column to df and fill per row. also find the max severity to save for custom severity metric
main_df["severity_tally"] = main_df.apply(calculate_severity, axis=1)
max_severity = main_df["severity_tally"].max()
main_df["custom_severity_score"] = (main_df["severity_tally"] / max_severity) * 10
main_df["custom_severity_score"] = (
    main_df["custom_severity_score"].clip(upper=10).round(1)
)


main_df.to_csv("data/dataset.csv", index=False)
