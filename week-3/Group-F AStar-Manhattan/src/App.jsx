//npm run dev

import { useState, useEffect, useCallback } from 'react';
import ReactFlow, { 
  Background, 
  Controls, 
  useNodesState,
  useEdgesState,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './App.css';

const nodeDefaults = {
  sourcePosition: 'right',
  targetPosition: 'left',
  draggable: true,
  style: {
    borderRadius: '50%',
    padding: '10px',
    minWidth: '100px',
    minHeight: '100px',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    aspectRatio: '1',
    fontSize: '16px',
  },
};

function App() {
  const [nodes, setNodes] = useState({});
  const [startNode, setStartNode] = useState('');
  const [goalNode, setGoalNode] = useState('');
  const [path, setPath] = useState(null);
  const [pathDepth, setPathDepth] = useState(null);
  
  const [flowNodes, setFlowNodes] = useNodesState([]);
  const [flowEdges, setFlowEdges] = useEdgesState([]);

  const [showPopup, setShowPopup] = useState(false);
  const [popupMessage, setPopupMessage] = useState('');

  const [nodePositions, setNodePositions] = useState({});

  const [newNodeData, setNewNodeData] = useState({
    name: '',
    position: [0, 0],
    heuristic: 0,
    connections: []
  });

  const [showNodeCreationModal, setShowNodeCreationModal] = useState(false);

  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [nodeToDelete, setNodeToDelete] = useState('');

  const NodeCreationModal = () => (
    <div className="modal-overlay">
      <div className="modal-content">
        <h3>Create New Node</h3>
        <div className="node-form">
          <div className="input-group">
            <label htmlFor="nodeName">Node Name:</label>
            <input
              id="nodeName"
              type="text"
              placeholder="e.g., A, B, C..."
              value={newNodeData.name}
              onChange={(e) => setNewNodeData({...newNodeData, name: e.target.value})}
            />
          </div>
          
          <div className="position-group">
            <label>Node Position:</label>
            <div className="position-inputs">
              <div className="input-group">
                <label htmlFor="xPos">X:</label>
                <input
                  id="xPos"
                  type="number"
                  placeholder="0"
                  value={newNodeData.position[0]}
                  onChange={(e) => setNewNodeData({
                    ...newNodeData, 
                    position: [parseInt(e.target.value) || 0, newNodeData.position[1]]
                  })}
                />
              </div>
              <div className="input-group">
                <label htmlFor="yPos">Y:</label>
                <input
                  id="yPos"
                  type="number"
                  placeholder="0"
                  value={newNodeData.position[1]}
                  onChange={(e) => setNewNodeData({
                    ...newNodeData, 
                    position: [newNodeData.position[0], parseInt(e.target.value) || 0]
                  })}
                />
              </div>
            </div>
          </div>
          
          <div className="input-group">
            <label htmlFor="heuristic">Heuristic Value:</label>
            <input
              id="heuristic"
              type="number"
              placeholder="e.g., 5"
              value={newNodeData.heuristic}
              onChange={(e) => setNewNodeData({...newNodeData, heuristic: parseInt(e.target.value) || 0})}
            />
          </div>
          
          <div className="input-group">
            <label htmlFor="connections">Connected Nodes:</label>
            <select
              id="connections"
              multiple
              value={newNodeData.connections}
              onChange={(e) => setNewNodeData({
                ...newNodeData,
                connections: Array.from(e.target.selectedOptions, option => option.value)
              })}
            >
              {Object.keys(nodes).map(node => (
                <option key={node} value={node}>{node}</option>
              ))}
            </select>
            <small className="help-text">Hold Ctrl/Cmd to select multiple nodes</small>
          </div>
          
          <div className="modal-buttons">
            <button 
              onClick={() => setShowNodeCreationModal(false)}
              className="cancel-button"
            >
              Cancel
            </button>
            <button 
              onClick={() => {
                handleCreateNode();
                setShowNodeCreationModal(false);
              }}
              disabled={!newNodeData.name}
              className="create-node-button"
            >
              Create Node
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const DeleteNodeModal = () => (
    <div className="modal-overlay">
      <div className="modal-content">
        <h3>Delete Node</h3>
        <div className="node-form">
          <div className="input-group">
            <label htmlFor="deleteNode">Select Node to Delete:</label>
            <select
              id="deleteNode"
              value={nodeToDelete}
              onChange={(e) => setNodeToDelete(e.target.value)}
            >
              <option value="">Select a node</option>
              {Object.keys(nodes).map(node => (
                <option key={node} value={node}>{node}</option>
              ))}
            </select>
          </div>
          
          <div className="modal-buttons">
            <button 
              onClick={() => {
                setShowDeleteModal(false);
                setNodeToDelete('');
              }}
              className="cancel-button"
            >
              Cancel
            </button>
            <button 
              onClick={() => {
                handleDeleteNode();
                setShowDeleteModal(false);
              }}
              disabled={!nodeToDelete}
              className="delete-button"
            >
              Delete Node
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const handleCreateNode = async () => {
    try {
      const response = await fetch('http://localhost:8000/add-node', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newNodeData),
      });
      
      if (response.ok) {
        setPopupMessage('Node created successfully!');
        setShowPopup(true);
        fetchNodes(); // Refresh the nodes list
        // Reset form
        setNewNodeData({
          name: '',
          position: [0, 0],
          heuristic: 0,
          connections: []
        });
      } else {
        setPopupMessage('Failed to create node');
        setShowPopup(true);
      }
    } catch (error) {
      console.error('Error creating node:', error);
      setPopupMessage('Error creating node');
      setShowPopup(true);
    }
  };

  const handleDeleteNode = async () => {
    try {
      const response = await fetch(`http://localhost:8000/delete-node/${nodeToDelete}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        setPopupMessage('Node deleted successfully!');
        setShowPopup(true);
        fetchNodes(); // Refresh the nodes list
        setNodeToDelete(''); // Reset selection
        
        // Reset start/goal nodes if the deleted node was selected
        if (startNode === nodeToDelete) setStartNode('');
        if (goalNode === nodeToDelete) setGoalNode('');
      } else {
        setPopupMessage('Failed to delete node');
        setShowPopup(true);
      }
    } catch (error) {
      console.error('Error deleting node:', error);
      setPopupMessage('Error deleting node');
      setShowPopup(true);
    }
  };

  useEffect(() => {
    fetchNodes();
  }, []);

  useEffect(() => {
    if (Object.keys(nodes).length === 0) return;

    const newFlowNodes = Object.entries(nodes).map(([nodeName, nodeData]) => ({
      ...nodeDefaults,
      id: nodeName,
      type: 'default',
      position: nodePositions[nodeName] || { 
        x: nodeData.position[1] * 150, 
        y: nodeData.position[0] * 150 
      },
      data: { 
        label: (
          <div style={{ textAlign: 'center', color: 'white' }}>
            <div>{nodeName}</div>
            <div>H: {nodeData.heuristic}</div>
          </div>
        )
      },
      style: {
        ...nodeDefaults.style,
        background: nodeName === startNode ? '#e63946' :    // Red for start node
                   nodeName === goalNode ? '#ff4d4d' :      // Light red for goal node
                   '#ff8c42',                               // Default orange for all other nodes
        border: '1px solid rgba(255, 255, 255, 0.2)',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
      },
    }));

    setFlowNodes(newFlowNodes);
  }, [nodes, startNode, goalNode, path, nodePositions]);

  useEffect(() => {
    if (Object.keys(nodes).length === 0) return;

    const processedConnections = new Set();
    const newFlowEdges = [];
    
    Object.entries(nodes).forEach(([nodeName, nodeData]) => {
      nodeData.connections?.forEach(connection => {
        const connectionKey = [nodeName, connection].sort().join('-');
        
        if (!processedConnections.has(connectionKey)) {
          const isInPath = path?.includes(nodeName) && path?.includes(connection);
          
          let source = nodeName;
          let target = connection;
          
          // Calculate Manhattan distance between nodes
          const distance = Math.abs(nodeData.position[0] - nodes[connection].position[0]) + 
                          Math.abs(nodeData.position[1] - nodes[connection].position[1]);
          
          if (path && path.includes(nodeName) && path.includes(connection)) {
            const nodeIndex = path.indexOf(nodeName);
            const connIndex = path.indexOf(connection);
            if (nodeIndex > connIndex) {
              [source, target] = [connection, nodeName];
            }
          }
          
          newFlowEdges.push({
            id: connectionKey,
            source: source,
            target: target,
            animated: isInPath,
            label: `${distance}`,  // Always show the Manhattan distance
            style: {
              stroke: '#ff7f50',
              strokeWidth: isInPath ? 3 : 1.5,
              opacity: isInPath ? 1 : 0.4,
            },
            labelStyle: { 
              fill: '#ff7f50',
              fontWeight: 'bold',
              fontSize: '20px',
              filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))',
            },
            labelBgStyle: { 
              fill: '#2d2e32',
              fillOpacity: 0.9,
              rx: 4,
              ry: 4,
            },
          });
          
          processedConnections.add(connectionKey);
        }
      });
    });

    setFlowEdges(newFlowEdges);
  }, [nodes, path]);

  const fetchNodes = async () => {
    try {
      const response = await fetch('http://localhost:8000/nodes');
      const data = await response.json();
      setNodes(data);
    } catch (error) {
      console.error('Error fetching nodes:', error);
    }
  };

  const findPath = async () => {
    try {
      const response = await fetch('http://localhost:8000/find-path', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          start_node: startNode,
          goal_node: goalNode,
        }),
      });
      const data = await response.json();
      setPath(data.path);
      
      // Update nodes with received heuristic values only after finding path
      setNodes(prevNodes => {
        const updatedNodes = {...prevNodes};
        Object.entries(updatedNodes).forEach(([nodeName, nodeData]) => {
          nodeData.heuristic = data.heuristics[nodeName];
        });
        return updatedNodes;
      });
      
      if (data.path) {
        let totalDepth = 0;
        for (let i = 0; i < data.path.length - 1; i++) {
          const currentNode = data.path[i];
          const nextNode = data.path[i + 1];
          const edge = flowEdges.find(edge => 
            (edge.source === currentNode && edge.target === nextNode) ||
            (edge.source === nextNode && edge.target === currentNode)
          );
          if (edge) {
            totalDepth += parseInt(edge.label);
          }
        }
        setPathDepth(totalDepth);
        setPopupMessage(
          <div>
            <div style={{ fontSize: '20px', marginBottom: '10px', color: '#646cff' }}>Path Found!</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
              Total Depth: {totalDepth}
            </div>
          </div>
        );
        setShowPopup(true);
      } else {
        setPopupMessage('No path found!');
        setShowPopup(true);
      }
    } catch (error) {
      console.error('Error finding path:', error);
      setPopupMessage('Error finding path');
      setShowPopup(true);
    }
  };

  const onNodeDrag = useCallback((event, node) => {
    setFlowNodes((nds) =>
      nds.map((n) => {
        if (n.id === node.id) {
          n.position = node.position;
        }
        return n;
      })
    );
  }, []);

  const onNodeDragStop = useCallback((event, node) => {
    setNodePositions(prev => ({
      ...prev,
      [node.id]: node.position
    }));
    
    setFlowNodes((nds) =>
      nds.map((n) => {
        if (n.id === node.id) {
          n.position = node.position;
        }
        return n;
      })
    );
  }, []);

  const CustomPopup = ({ message, onClose }) => (
    <div className="popup-overlay">
      <div className="popup-content">
        <div>{message}</div>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );

  return (
    <div className="flow-wrapper">
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        onNodeDrag={onNodeDrag}
        onNodeDragStop={onNodeDragStop}
        fitView
        defaultViewport={{ x: 0, y: 0, zoom: 1.5 }}
      >
        <Background />
        <Controls />
        {showPopup && (
          <CustomPopup 
            message={popupMessage} 
            onClose={() => setShowPopup(false)} 
          />
        )}
        
        <div className="panels-container">
          <Panel position="center-bottom" className="control-panel">
            <div className="select-group">
              <select 
                value={startNode} 
                onChange={(e) => setStartNode(e.target.value)}
              >
                <option value="">Select Start Node</option>
                {Object.keys(nodes).map(node => (
                  <option key={node} value={node}>{node}</option>
                ))}
              </select>

              <select 
                value={goalNode} 
                onChange={(e) => setGoalNode(e.target.value)}
              >
                <option value="">Select Goal Node</option>
                {Object.keys(nodes).map(node => (
                  <option key={node} value={node}>{node}</option>
                ))}
              </select>
            </div>

            <button 
              onClick={findPath} 
              disabled={!startNode || !goalNode}
              className="find-path-button"
            >
              Find Path
            </button>
            
            <button 
              onClick={() => setShowNodeCreationModal(true)}
              className="create-node-button"
            >
              Create New Node
            </button>
            
            <button 
              onClick={() => setShowDeleteModal(true)}
              className="delete-node-button"
            >
              Delete Node
            </button>
            
            {pathDepth !== null && (
              <div style={{ marginTop: '10px', fontWeight: 'bold' }}>
                Path Depth: {pathDepth} steps
              </div>
            )}
          </Panel>
        </div>
      </ReactFlow>

      {showNodeCreationModal && <NodeCreationModal />}
      {showDeleteModal && <DeleteNodeModal />}
    </div>
  );
}

export default App;
