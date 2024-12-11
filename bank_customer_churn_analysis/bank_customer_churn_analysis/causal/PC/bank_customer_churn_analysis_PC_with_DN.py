from causallearn.graph.Edge import Edge
from causallearn.graph.Node import Node
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
import networkx as nx
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
# Load the dataset
file_path = 'https://raw.githubusercontent.com/serterergun/Implementation/main/bank_customer_churn_analysis/data/bank_customer_churn_dataset.csv'
df = pd.read_csv(file_path)

# Drop specified columns
df.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)

# Encode categorical columns
df = pd.get_dummies(df, columns=['Gender','Geography'], drop_first=True)

# Ensure all columns are numeric and fill any missing values
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True) #aaaaaaaa

# Convert all columns to float64 to ensure compatibility with np.isnan
df = df.astype(np.float64)

# Get labels and convert data to numpy array
labels = df.columns.tolist()
data = df.to_numpy()

def apply_domain_knowledge(cg):
  cg.G.add_directed_edge(GraphNode("Age"), GraphNode("EstimatedSalary"))
  cg.G.add_directed_edge(GraphNode("Age"), GraphNode("CreditScore"))
  cg.G.add_directed_edge(GraphNode("Tenure"), GraphNode("CreditScore"))
  cg.G.add_directed_edge(GraphNode("Tenure"), GraphNode("NumOfProducts"))
  cg.G.add_directed_edge(GraphNode("Tenure"), GraphNode("IsActiveMember"))
  cg.G.add_directed_edge(GraphNode("HasCrCard"), GraphNode("Exited"))
  cg.G.add_directed_edge(GraphNode("Tenure"), GraphNode("Exited"))
  cg.G.add_directed_edge(GraphNode("EstimatedSalary"), GraphNode("Exited"))

  return cg

cg = pc(data, node_names=labels)
cg = apply_domain_knowledge(cg)

pyd = GraphUtils.to_pydot(cg.G, labels=labels)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.figure(figsize=(50, 50))
plt.axis('off')
plt.imshow(img)
plt.show()



