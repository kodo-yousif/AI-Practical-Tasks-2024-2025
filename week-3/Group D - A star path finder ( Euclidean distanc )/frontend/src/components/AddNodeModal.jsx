import React, { useState } from 'react';
import './AddNodeModal.css';

const AddNodeModal = ({ onAdd, onClose, existingNames, showNotification }) => {
  const [nodeName, setNodeName] = useState('');
  const [nodeValue, setNodeValue] = useState('');
  const [error, setError] = useState('');

  const handleNameChange = (e) => {
    const newName = e.target.value;
    setNodeName(newName);
    
    if (existingNames.some(name => name.toLowerCase() === newName.toLowerCase())) {
      setError('This node name already exists');
    } else {
      setError('');
    }
  };

  const handleAdd = () => {
    if (!nodeName || !nodeValue) {
      setError('Both fields are required.');
      return;
    }

    if (existingNames.some(name => name.toLowerCase() === nodeName.toLowerCase())) {
      setError('Node name must be unique.');
      return;
    }

    onAdd(nodeName, nodeValue);
    showNotification(`Node "${nodeName}" added successfully!`, 'success');
    onClose();
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <button className="modal-close-btn" onClick={onClose}>×</button>
        <div className="modal-header">
          <h3>Add New Node</h3>
          <p className="modal-subtitle">Create a new node in the graph</p>
        </div>
        <div className="modal-body">
          <div className="input-group">
            <label>Node Name</label>
            <input
              type="text"
              placeholder="Enter node name"
              value={nodeName}
              onChange={handleNameChange}
              className={error && error.includes('name') ? 'input-error' : ''}
            />
          </div>
          <div className="input-group">
            <label>Node Value</label>
            <input
              type="number"
              placeholder="Enter node value"
              value={nodeValue}
              onChange={(e) => setNodeValue(e.target.value)}
              className={error && error.includes('value') ? 'input-error' : ''}
            />
          </div>
          {error && <div className="error-message">{error}</div>}
        </div>
        <div className="modal-footer">
          <button className="modal-cancel-btn" onClick={onClose}>Cancel</button>
          <button className="modal-add-btn" onClick={handleAdd}>Add Node</button>
        </div>
      </div>
    </div>
  );
};

export default AddNodeModal; 