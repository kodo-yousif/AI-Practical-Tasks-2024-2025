import { useState } from "react";
import ReactFlow, {
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  Controls,
  Background,
} from "reactflow";
import "reactflow/dist/style.css";
import "./App.css";
import PathVisualization from './components/PathVisualization';
import PathNotification from './components/PathNotification';
import GlobalNotification from './components/GlobalNotification';
import AddNodeModal from './components/AddNodeModal';
import EdgeCostModal from './components/EdgeCostModal';

const initialNodes = [
  {
    id: "1",
    data: { label: "(ID: 1) -> Node S - Value: 17" },
    position: { x: 100, y: 100 },
    draggable: true,
  },
  {
    id: "2",
    data: { label: "(ID: 2) -> Node C - Value: 4" },
    position: { x: 300, y: 50 },
    draggable: true,
  },
  {
    id: "3",
    data: { label: "(ID: 3) -> Node A - Value: 10" },
    position: { x: 200, y: 300 },
    draggable: true,
  },
  {
    id: "4",
    data: { label: "(ID: 4) -> Node B - Value: 13" },
    position: { x: 400, y: 150 },
    draggable: true,
  },
  {
    id: "5",
    data: { label: "(ID: 5) -> Node D - Value: 2" },
    position: { x: 500, y: 50 },
    draggable: true,
  },
  {
    id: "6",
    data: { label: "(ID: 6) -> Node F - Value: 1" },
    position: { x: 600, y: 250 },
    draggable: true,
  },
  {
    id: "7",
    data: { label: "(ID: 7) -> Node E - Value: 4" },
    position: { x: 400, y: 350 },
    draggable: true,
  },
  {
    id: "8",
    data: { label: "(ID: 8) -> Node G - Value: 0" },
    position: { x: 700, y: 400 },
    draggable: true,
  },
];

const initialEdges = [
  { id: "e1-3", source: "1", target: "3", animated: true, label: "Cost: 6", data: { cost: 6 } },
  { id: "e1-2", source: "1", target: "2", animated: true, label: "Cost: 10", data: { cost: 10 } },
  { id: "e1-4", source: "1", target: "4", animated: true, label: "Cost: 5", data: { cost: 5 } },
  { id: "e5-2", source: "5", target: "2", animated: true, label: "Cost: 6", data: { cost: 6 } },
  { id: "e5-4", source: "5", target: "4", animated: true, label: "Cost: 7", data: { cost: 7 } },
  { id: "e5-6", source: "5", target: "6", animated: true, label: "Cost: 6", data: { cost: 6 } },
  { id: "e6-7", source: "6", target: "7", animated: true, label: "Cost: 4", data: { cost: 4 } },
  { id: "e7-3", source: "7", target: "3", animated: true, label: "Cost: 6", data: { cost: 6 } },
  { id: "e4-7", source: "4", target: "7", animated: true, label: "Cost: 6", data: { cost: 6 } },
  { id: "e6-8", source: "6", target: "8", animated: true, label: "Cost: 3", data: { cost: 3 } },
];

function getNodeColor(nodeId, pathNodes, startNode, endNode, isSelected) {
  let backgroundColor = '#fff';  
  
  if (nodeId === startNode) {
    backgroundColor = '#90EE90';  
  } else if (nodeId === endNode) {
    backgroundColor = '#FF5252';  
  } else if (pathNodes?.includes(nodeId)) {
    backgroundColor = '#87CEFA';  
  }

  return {
    backgroundColor,
    border: isSelected ? '2px solid #FFD700' : '1px solid #666',
    boxShadow: isSelected ? '0 0 0 1px rgba(255, 215, 0, 0.3)' : 'none',
    cursor: 'pointer'
  };
}

function Graph() {
  const [nodes, setNodes] = useState(initialNodes.map(node => ({
    ...node,
    style: { cursor: 'pointer' }  
  })));
  const [edges, setEdges] = useState(initialEdges);
  const [startNode, setStartNode] = useState(null);
  const [endNode, setEndNode] = useState(null);
  const [path, setPath] = useState([]);
  const [steps, setSteps] = useState([]);
  const [showVisualization, setShowVisualization] = useState(false);
  const [showPathNotification, setShowPathNotification] = useState(false);
  const [pathInfo, setPathInfo] = useState(null);
  const [notification, setNotification] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [isPathFound, setIsPathFound] = useState(false);
  const [showAddNodeModal, setShowAddNodeModal] = useState(false);
  const [showEdgeCostModal, setShowEdgeCostModal] = useState(false);
  const [pendingEdge, setPendingEdge] = useState(null);

  const onConnect = (params) => {
    setPendingEdge(params);
    setShowEdgeCostModal(true);
  };

  const handleAddEdgeCost = (costValue) => {
    const newEdge = {
      ...pendingEdge,
      data: { cost: costValue },
      label: `Cost: ${costValue}`,
      animated: true,
    };
    setEdges((eds) => addEdge(newEdge, eds));
    setPendingEdge(null);
    showNotification("Edge added successfully!", "success");
  };

  const onAddNode = (nodeName, nodeValue) => {
    const newNodeId = (nodes.length + 1).toString();
    const newNode = {
      id: newNodeId,
      data: {
        label: `(ID: ${newNodeId}) -> Node ${nodeName} - Value: ${parseInt(
          nodeValue
        )} `,
      },
      position: { x: Math.random() * 600, y: Math.random() * 400 },
      draggable: true,
      style: getNodeColor(
        newNodeId,
        path,
        startNode,
        endNode,
        false
      )
    };
    setNodes((nds) => [...nds, newNode]);
  };



  const showNotification = (message, type = 'info') => {
    setNotification({ message, type });
  };

  const handleReset = () => {
    setNodes((nds) => 
      nds.map(node => ({
        ...node,
        style: getNodeColor(  
          node.id,
          null,    
          null,    
          null,    
          false    
        )
      }))
    );
    
    setEdges((eds) =>
      eds.map(edge => ({
        ...edge,
        style: {
          stroke: '#b1b1b7',
          strokeWidth: 1
        },
        animated: true
      }))
    );

    setStartNode(null);
    setEndNode(null);
    setPath([]);
    setSteps([]);  
    setSelectedNode(null);
    setIsPathFound(false);
    setShowVisualization(false);  
    showNotification("Graph reset to initial state", "info");
  };

  async function onFindPath() {
    if (!startNode || !endNode) {
      showNotification("Please select both start and end nodes first!", "warning");
      return;
    }

    const graphData = {
      nodes: nodes.map((node) => {
        const match = node.data.label.match(/\(ID: (\d+)\) -> Node (\w+) - Value: (\d+)/);
        if (!match) {
          console.error("Invalid node label format:", node.data.label);
          return null;
        }
        const [_, idPart, namePart, valuePart] = match;
        return {
          id: node.id,
          name: namePart,
          value: parseInt(valuePart),
          x: node.position.x,
          y: node.position.y,
        };
      }).filter(Boolean),
      edges: edges.map((edge) => ({
        source: edge.source,
        target: edge.target,
        cost: parseInt(edge.data?.cost) || 0,
      })),
      startNode,
      endNode,
    };

    try {
      const response = await fetch("http://localhost:8000/find-path", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(graphData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to find path");
      }

      const data = await response.json();
      console.log("Complete A* Search Result:", data);
      
      setPath(data.path_nodes); 
      
      setNodes((nds) =>
        nds.map((node) => ({
          ...node,
          style: getNodeColor(
            node.id,
            data.path_nodes,
            startNode,
            endNode,
            node.id === selectedNode?.id
          ),
        }))
      );

      setEdges((eds) =>
        eds.map((edge) => {
          const pathNodes = data.path_nodes;
          let isPathEdge = false;
          let isCorrectDirection = false;

          for (let i = 0; i < pathNodes.length - 1; i++) {
            const currentNodeId = pathNodes[i];
            const nextNodeId = pathNodes[i + 1];
            
            if (edge.source === currentNodeId && edge.target === nextNodeId) {
              isPathEdge = true;
              isCorrectDirection = true;
              break;
            } else if (edge.source === nextNodeId && edge.target === currentNodeId) {
              isPathEdge = true;
              isCorrectDirection = false;
              break;
            }
          }

          if (isPathEdge) {
            if (!isCorrectDirection) {
              return {
                ...edge,
                source: edge.target,
                target: edge.source,
                style: {
                  ...edge.style,
                  stroke: '#FFA500',
                  strokeWidth: 3,
                },
                animated: true,
              };
            } else {
              return {
                ...edge,
                style: {
                  ...edge.style,
                  stroke: '#FFA500',
                  strokeWidth: 3,
                },
                animated: true,
              };
            }
          }

          return {
            ...edge,
            style: {
              ...edge.style,
              stroke: '#b1b1b7',
              strokeWidth: 1,
            },
            animated: false,
          };
        })
      );

      const pathIds = data.path.map(([id, _]) => id).join(" => ");
      const pathNames = data.path.map(([_, name]) => name).join(" => ");

      setPathInfo({
        pathIds,
        pathNames,
        totalCost: data.total_cost
      });
      setShowPathNotification(true);

      setSteps(data.steps);

      setIsPathFound(true);
      showNotification("Path found! Graph is now locked. Click Reset to make changes.", "success");

    } catch (error) {
      console.error("Error finding path:", error);
      showNotification(error.message || "Error finding path", "error");
    }
  }

  function visualizeSteps() {
    if (steps.length === 0) {
      showNotification("Please find a path first!", "warning");
      return;
    }
    setShowVisualization(true);
  }

  const onNodeClick = (event, node) => {
    event.stopPropagation();  
    setSelectedNode(node);
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        style: getNodeColor(
          n.id,
          path, 
          startNode,
          endNode,
          n.id === node.id
        ),
      }))
    );
  };

  const onSetAsStart = () => {
    if (!selectedNode) {
      showNotification("Please select a node first!", "warning");
      return;
    }
    setStartNode(selectedNode.id);
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        style: getNodeColor(node.id, path, selectedNode.id, endNode, node.id === selectedNode.id)
      }))
    );
    showNotification(`Node ${selectedNode.id} set as start node`, "success");
  };

  const onSetAsEnd = () => {
    if (!selectedNode) {
      showNotification("Please select a node first!", "warning");
      return;
    }
    setEndNode(selectedNode.id);
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        style: getNodeColor(node.id, path, startNode, selectedNode.id, node.id === selectedNode.id)
      }))
    );
    showNotification(`Node ${selectedNode.id} set as end node`, "success");
  };

  const onDeleteSelected = () => {
    if (!selectedNode) {
      showNotification("Please select a node first!", "warning");
      return;
    }
    setNodes((nds) => nds.filter((node) => node.id !== selectedNode.id));
    setEdges((eds) =>
      eds.filter((edge) => edge.source !== selectedNode.id && edge.target !== selectedNode.id)
    );
    setSelectedNode(null);
    showNotification(`Node ${selectedNode.id} deleted`, "success");
  };

  const onPaneClick = () => {
    setSelectedNode(null);
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        style: getNodeColor(n.id, path, startNode, endNode, false),
      }))
    );
  };

  return (
    <div style={{ height: "100vh", width: "100vw", overflow: "hidden" }}>
      <div className="flex justify-around items-center">
        <button 
          onClick={() => setShowAddNodeModal(true)}
          style={{ 
            marginBottom: "10px",
            backgroundColor: isPathFound ? '#666' : undefined
          }}
          disabled={isPathFound}
        >
          Add Node
        </button>
        <button 
          onClick={onSetAsStart}
          style={{ 
            marginBottom: "10px",
            backgroundColor: (!selectedNode || isPathFound) ? '#666' : undefined
          }}
          disabled={!selectedNode || isPathFound}
        >
          Set as Start
        </button>
        <button 
          onClick={onSetAsEnd}
          style={{ 
            marginBottom: "10px",
            backgroundColor: (!selectedNode || isPathFound) ? '#666' : undefined
          }}
          disabled={!selectedNode || isPathFound}
        >
          Set as End
        </button>
        <button 
          onClick={onDeleteSelected}
          style={{ 
            marginBottom: "10px",
            backgroundColor: (!selectedNode || isPathFound) ? '#666' : undefined
          }}
          disabled={!selectedNode || isPathFound}
        >
          Delete Selected
        </button>
        <button 
          onClick={onFindPath} 
          style={{ 
            marginBottom: "10px",
            backgroundColor: isPathFound ? '#666' : '#ffffff',  
            color: isPathFound ? '#fff' : '#000000',  
            boxShadow: isPathFound ? 'none' : '0 0 10px rgba(255, 255, 255, 0.3)',  
            border: isPathFound ? 'none' : '1px solid rgba(255, 255, 255, 0.2)'  
          }}
          disabled={isPathFound}
        >
          Find Path
        </button>
        <button 
          onClick={visualizeSteps} 
          style={{ 
            marginBottom: "10px",
            animation: isPathFound ? 'pulseButton 2s infinite' : 'none',
            border: isPathFound ? '1px solid #f7a531' : 'none'  
          }}
        >
          Iteration Steps
        </button>
        {isPathFound && (
          <button 
            onClick={handleReset}
            style={{ 
              marginBottom: "10px",
              backgroundColor: '#D32F2F'  
            }}
          >
            Reset Graph
          </button>
        )}
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onConnect={onConnect}
        onNodesChange={(changes) =>
          setNodes((nds) => applyNodeChanges(changes, nds))
        }
        onEdgesChange={(changes) =>
          setEdges((eds) => applyEdgeChanges(changes, eds))
        }
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}  
        style={{ width: "100%", height: "90%" }}
      >
        <Controls />
        <Background color="#aaa" gap={16} />
      </ReactFlow>
      {showAddNodeModal && (
        <AddNodeModal 
          onAdd={onAddNode}
          onClose={() => setShowAddNodeModal(false)}
          existingNames={nodes.map(node => {
            const match = node.data.label.match(/Node (\w+)/);
            return match ? match[1] : '';
          })}
          showNotification={showNotification}
        />
      )}
      {showPathNotification && (
        <PathNotification 
          pathInfo={pathInfo}
          onClose={() => setShowPathNotification(false)}
        />
      )}
      {showVisualization && (
        <PathVisualization 
          steps={steps} 
          onClose={() => setShowVisualization(false)} 
        />
      )}
      {notification && (
        <GlobalNotification
          message={notification.message}
          type={notification.type}
          onClose={() => setNotification(null)}
        />
      )}
      {showEdgeCostModal && (
        <EdgeCostModal
          onAdd={handleAddEdgeCost}
          onClose={() => {
            setShowEdgeCostModal(false);
            setPendingEdge(null);
          }}
        />
      )}
    </div>
  );
}

function App() {
  return (
    <div className="App">
      <Graph />
    </div>
  );
}

export default App;
