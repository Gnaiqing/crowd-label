from crowdkit.aggregation.dawid_skene import DawidSkene
from crowdkit.aggregation.majority_vote import MajorityVote
import pandas as pd


def load_data(path):
    df = pd.read_csv(path,
                     sep="\t",
                     usecols=["INPUT:url",
                              "INPUT:query",
                              "OUTPUT:result",
                              "GOLDEN:result",
                              "ASSIGNMENT:task_id",
                              "ASSIGNMENT:worker_id"])
    df = df.rename(columns={
        "INPUT:url":"url",
        "INPUT:query":"query",
        "OUTPUT:result":"label",
        "GOLDEN:result":"golden",
        "ASSIGNMENT:task_id":"task",
        "ASSIGNMENT:worker_id":"performer",
    })
    return df

def describe_data(df, label):
    print("Dataframe ",label)
    print("Size :", len(df))
    print("Num. tasks:", len(pd.unique(df["task"])))
    print("Num. workers:", len(pd.unique(df["performer"])))

df = load_data("data/crowdlabel-raw.tsv")
df_val = df[df["golden"].isna() == False] # the validation set (trap questions) used to evaluate user ability
df_test = df[df["golden"].isna()] # the test set (main pool) that has no available ground-truth label
describe_data(df_val, "Valid pool")
describe_data(df_test, "Main pool")
aggregation_method = "DS" # one of "MV"(majority vote), "DS"(Dawid-Skene)
if aggregation_method == "MV":
    aggregated_labels = MajorityVote().fit_predict(df_test)

elif aggregation_method == "DS":
    aggregated_labels = DawidSkene(n_iter=100).fit_predict(df_test)

df_aggregated = df_test.drop_duplicates(subset="task").set_index("task")
df_aggregated["relevance"] = aggregated_labels
df_aggregated = df_aggregated[["url","query","relevance"]]
# replace values for scoring purpose
df_aggregated["relevance"].replace(to_replace="Relevant", value="RELEVANT_PLUS",inplace=True)
df_aggregated["relevance"].replace(to_replace="RelevantMinus", value="RELEVANT_MINUS",inplace=True)
df_aggregated["relevance"].replace(to_replace="Irrelevant", value="IRRELEVANT",inplace=True)
output_file = "aggregated_%s.tsv" % aggregation_method
df_aggregated.to_csv(output_file,sep="\t",index=False)

