import pandas as pd
import json

amsterdam = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/AmsterdamTrees.csv"
)
la = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/LosAngelesTrees.csv"
)
seattle = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/SeattleTrees.csv"
)
montreal = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/MontrealTrees.csv"
)
ny = pd.read_csv("/shared/projects/autoarborist/data/tree_locations/NewYorkTrees.csv")
boulder = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/BoulderTrees.csv"
)
buffalo = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/BuffaloTrees.csv"
)
calgary = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/CalgaryTrees.csv"
)
columbus = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/ColumbusTrees.csv"
)
denver = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/DenverTrees.csv"
)
kitch = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/KitchenerTrees.csv"
)
edmonton = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/EdmontonTrees.csv"
)
sf = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/SanFranciscoTrees.csv"
)
sj = pd.read_csv("/shared/projects/autoarborist/data/tree_locations/SanJoseTrees.csv")
sioux = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/SiouxFallsTrees.csv"
)
surrey = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/SurreyTrees.csv"
)
dc = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/WashingtonDcTrees.csv"
)
van = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/VancouverTrees.csv"
)
pitts = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/PittsburghTrees.csv"
)
bloom = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/BloomingtonTrees.csv"
)
cambridge = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/CambridgeTrees.csv"
)
charlottesville = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/CharlottesvilleTrees.csv"
)
cupertino = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/CupertinoTrees.csv"
)
sm = pd.read_csv(
    "/shared/projects/autoarborist/data/tree_locations/SantaMonicaTrees.csv"
)
combined_genus = pd.concat([amsterdam["GENUS"], la["GENUS"]])
combined_genus = pd.concat([combined_genus, seattle["GENUS"]])
combined_genus = pd.concat([combined_genus, montreal["GENUS"]])
combined_genus = pd.concat([combined_genus, ny["GENUS"]])
combined_genus = pd.concat([combined_genus, boulder["GENUS"]])
combined_genus = pd.concat([combined_genus, buffalo["GENUS"]])
combined_genus = pd.concat([combined_genus, buffalo["GENUS"]])
combined_genus = pd.concat([combined_genus, calgary["GENUS"]])
combined_genus = pd.concat([combined_genus, columbus["GENUS"]])
combined_genus = pd.concat([combined_genus, denver["GENUS"]])
combined_genus = pd.concat([combined_genus, edmonton["GENUS"]])
combined_genus = pd.concat([combined_genus, kitch["GENUS"]])
combined_genus = pd.concat([combined_genus, sf["GENUS"]])
combined_genus = pd.concat([combined_genus, sj["GENUS"]])
combined_genus = pd.concat([combined_genus, sioux["GENUS"]])
combined_genus = pd.concat([combined_genus, surrey["GENUS"]])
combined_genus = pd.concat([combined_genus, dc["GENUS"]])
combined_genus = pd.concat([combined_genus, van["GENUS"]])
combined_genus = pd.concat([combined_genus, pitts["GENUS"]])
combined_genus = pd.concat([combined_genus, bloom["GENUS"]])
combined_genus = pd.concat([combined_genus, cambridge["GENUS"]])
combined_genus = pd.concat([combined_genus, charlottesville["GENUS"]])
combined_genus = pd.concat([combined_genus, cupertino["GENUS"]])
combined_genus = pd.concat([combined_genus, sm["GENUS"]])
final_list = unique_genus = combined_genus.unique().tolist()
print(len(final_list))
sorted_final_list = sorted(final_list)
df_sorted_genus = pd.DataFrame(sorted_final_list, columns=["Genus"])
sorted_final_list


def list_to_dict_with_indices(input_list):
    input_list = sorted(input_list)
    return {item: index for index, item in enumerate(input_list)}


class_to_idx = list_to_dict_with_indices(sorted_final_list)
with open("config/tree/config.json", "w") as w:
    json.dump(class_to_idx, w)
