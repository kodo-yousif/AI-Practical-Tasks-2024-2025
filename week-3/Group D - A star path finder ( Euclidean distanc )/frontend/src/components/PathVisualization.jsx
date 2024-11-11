import React, { useState, useRef } from 'react';
import ReactFlow, { 
  MiniMap, 
  Controls, 
  Background,
  getStraightPath,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './PathVisualization.css';

const CustomEdge = ({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style = {},
  markerEnd,
}) => {
  const [edgePath] = getStraightPath({
    sourceX,
    sourceY: sourceY + 100,
    targetX,
    targetY: targetY - 20,
  });

  return (
    <>
      <path
        id={id}
        className="animated-path"
        d={edgePath}
        strokeWidth={3}
        stroke="#fff"
        fill="none"
        strokeDasharray="5,5"
      />
      <marker
        id="arrowhead"
        viewBox="0 0 10 10"
        refX="8"
        refY="5"
        markerWidth="8"
        markerHeight="8"
        orient="auto"
      >
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#fff" />
      </marker>
    </>
  );
};

const PathVisualization = ({ steps, onClose }) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const isLastStep = currentStepIndex === steps.length - 1;
  const [showOutput, setShowOutput] = useState(false);
  const reactFlowInstance = useRef(null);

  const handleNext = () => {
    if (currentStepIndex < steps.length - 1) {
      const nextIndex = currentStepIndex + 1;
      setCurrentStepIndex(nextIndex);
      
      setTimeout(() => {
        const nextNodePosition = {
          x: 300,
          y: nextIndex * 800
        };
        
        const nodeContent = steps[nextIndex];
        const hasLongContent = nodeContent.neighbors_info?.length > 2;
        const hasVeryLongContent = nodeContent.neighbors_info?.length > 4;
        let zoomLevel;
        let yOffset;

        if (hasVeryLongContent) {
          zoomLevel = 0.4;
          yOffset = 300;
        } else if (hasLongContent) {
          zoomLevel = 0.5;
          yOffset = 250;
        } else {
          zoomLevel = 0.6;
          yOffset = 200;
        }
        
        reactFlowInstance.current?.setCenter(
          nextNodePosition.x + 225,
          nextNodePosition.y + yOffset,
          { 
            duration: 1000,
            zoom: zoomLevel
          }
        );
      }, 100);
    } else if (isLastStep && !showOutput) {
      setShowOutput(true);
    }
  };

  const handlePrevious = () => {
    if (showOutput) {
      setShowOutput(false);
      return;
    }
    if (currentStepIndex > 0) {
      setCurrentStepIndex(prev => prev - 1);
      
      setTimeout(() => {
        const currentNode = stepNodes[currentStepIndex - 1];
        const nodeContent = steps[currentStepIndex - 1];
        const hasLongContent = nodeContent.neighbors_info?.length > 2;
        const hasVeryLongContent = nodeContent.neighbors_info?.length > 4;
        let zoomLevel;
        let yOffset;

        if (hasVeryLongContent) {
          zoomLevel = 0.4;
          yOffset = 300;
        } else if (hasLongContent) {
          zoomLevel = 0.5;
          yOffset = 250;
        } else {
          zoomLevel = 0.6;
          yOffset = 200;
        }

        reactFlowInstance.current?.setCenter(
          currentNode.position.x + 225,
          currentNode.position.y + yOffset,
          { 
            duration: 1000,
            zoom: zoomLevel
          }
        );
      }, 100);
    }
  };

  if (!steps || steps.length === 0) return null;

  const visibleSteps = steps.slice(0, currentStepIndex + 1);

  const stepNodes = visibleSteps.map((step, index) => ({
    id: `step-${step.step}`,
    type: 'stepNode',
    position: { 
      x: 300,
      y: index * 800
    },
    data: {
      ...step,
      status: getNodeStatus(step)
    }
  }));

  const stepEdges = visibleSteps.slice(0, -1).map((_, index) => ({
    id: `edge-${index}`,
    source: `step-${visibleSteps[index].step}`,
    target: `step-${visibleSteps[index + 1].step}`,
    type: 'custom',
    animated: true,
    markerEnd: 'arrowhead',
  }));

  const edgeTypes = {
    custom: CustomEdge,
  };

  function getNodeStatus(step) {
    if (step.message.includes('Start node')) {
      return 'start';
    } else if (step.neighbors_info && step.neighbors_info.length > 0) {
      return 'exploring';
    } else if (step.is_goal) {
      return 'goal';
    } else {
      return 'processing';
    }
  }

  const StepNode = ({ data }) => {
    const statusColors = {
      start: {
        bg: '#E8F5E9',
        border: '#66BB6A',
        text: '#2E7D32'
      },
      exploring: {
        bg: '#E3F2FD',
        border: '#42A5F5',
        text: '#1565C0'
      },
      processing: {
        bg: '#FFF3E0',
        border: '#FFA726',
        text: '#E65100'
      },
      goal: {
        bg: '#FCE4EC',
        border: '#EC407A',
        text: '#C2185B'
      }
    };

    const colors = statusColors[data.status] || statusColors.processing;

    return (
      <div 
        className="step-node-content"
        style={{ 
          backgroundColor: colors.bg,
          borderColor: colors.border,
          borderWidth: '2px',
          borderStyle: 'solid',
          color: '#212121',
          boxShadow: `0 4px 8px ${colors.border}40`
        }}
      >
        <div className="step-header" style={{ color: colors.text }}>
          Iteration {data.step}
        </div>
        <div className="step-details">
          <div className="current-node-section">
            <strong>Current Node:</strong><br />
            {data.current_name} (f = {data.current_f?.toFixed(2)})
          </div>

          {data.neighbors_info && data.neighbors_info.length > 0 && (
            <div className="neighbors-section">
              <strong>Neighbors of {data.current_name}:</strong>
              {data.neighbors_info.map((neighbor, idx) => (
                <div key={idx} className="neighbor-info">
                  <div className="neighbor-name">{neighbor.name}:</div>
                  <div className="neighbor-calculations">
                    g({neighbor.name}) = {neighbor.g?.toFixed(2)}<br />
                    h({neighbor.name}) = {neighbor.h?.toFixed(2)}<br />
                    f({neighbor.name}) = {neighbor.g?.toFixed(2)} + {neighbor.h?.toFixed(2)} = {neighbor.f?.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="lists-section">
            <div className="open-list">
              <strong>Open List:</strong> [
              {data.open_set_details?.length > 0 
                ? data.open_set_details.map(([name, f]) => 
                    `${name}(f=${f?.toFixed(2)})`
                  ).join(', ')
                : ''
              }]
            </div>
            <div className="closed-list">
              <strong>Closed List:</strong> [{
                data.closed_set?.map(id => {
                  const node = steps.find(s => s.current === id);
                  return node ? node.current_name : id;
                }).join(', ')
              }]
            </div>
          </div>

          {data.is_goal && (
            <div className="goal-reached">
              Goal Node Reached
            </div>
          )}
        </div>
      </div>
    );
  };

  const nodeTypes = {
    stepNode: StepNode
  };

  return (
    <div className="visualization-overlay">
      <div className="visualization-container">
        <button className="close-button" onClick={onClose}>×</button>
        <h2>Iteration Steps</h2>
        
        <div className="visualization-content">
          {!showOutput ? (
            <div className="flow-container">
              <ReactFlow
                nodes={stepNodes}
                edges={stepEdges}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                fitView
                minZoom={0.1}
                maxZoom={2}
                defaultViewport={{ x: 0, y: 0, zoom: 0.5 }}
                onInit={(instance) => {
                  reactFlowInstance.current = instance;
                  instance.fitView({ padding: 0.3 });
                }}
                fitViewOptions={{
                  padding: 0.3,
                  duration: 800
                }}
              >
                <Controls />
                <Background color="#333" gap={16} />
              </ReactFlow>
            </div>
          ) : (
            <div className="output-container">
              <h3>Final Path Found</h3>
              <div className="path-details">
                <p><strong>Path:</strong> {
                  steps.find(step => step.is_goal)?.path?.map(([_, name]) => name).join(" => ") || "No path found"
                }</p>
                <p><strong>Total Cost:</strong> {
                  steps.find(step => step.is_goal)?.total_cost?.toFixed(2) || "N/A"
                }</p>
              </div>
            </div>
          )}
        </div>

        <div className="fixed-navigation">
          <button 
            onClick={handlePrevious}
            disabled={currentStepIndex === 0 && !showOutput}
            className="nav-button"
          >
            Previous
          </button>
          <span className="step-counter">
            {showOutput ? 'Final Path' : `Step ${currentStepIndex + 1} of ${steps.length}`}
          </span>
          <button 
            onClick={handleNext}
            disabled={isLastStep && showOutput}
            className="nav-button"
          >
            {isLastStep && !showOutput ? 'Show Path' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default PathVisualization;