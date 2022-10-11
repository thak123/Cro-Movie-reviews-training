import pandas as pd

def downsample(df: pd.DataFrame, label_col_name: str) -> pd.DataFrame:
    # find the number of observations in the smallest group
    nmin = df[label_col_name].value_counts().min()
    return (
        df
        # split the dataframe per group
        .groupby(label_col_name)
        # sample nmin observations from each group
        .apply(lambda x: x.sample(nmin))
        # recombine the dataframes
        .reset_index(drop=True)
    )


def oversample(df, label_col_name: str):
    classes = df.label.value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df[label_col_name] == key])
    classes_sample = []
    for i in range(1, len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe, classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df