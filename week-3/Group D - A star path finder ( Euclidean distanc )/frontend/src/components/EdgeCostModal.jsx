import React, { useState } from 'react';
import './EdgeCostModal.css';

const EdgeCostModal = ({ onAdd, onClose }) => {
  const [cost, setCost] = useState('');
  const [error, setError] = useState('');

  const handleAdd = () => {
    if (!cost) {
      setError('Cost is required.');
      return;
    }
    
    const costValue = parseInt(cost);
    if (isNaN(costValue) || costValue < 0) {
      setError('Please enter a valid positive number.');
      return;
    }

    onAdd(costValue);
    onClose();
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <button className="modal-close-btn" onClick={onClose}>×</button>
        <div className="modal-header">
          <h3>Add Edge Cost</h3>
          <p className="modal-subtitle">Set the cost for this connection</p>
        </div>
        <div className="modal-body">
          <div className="input-group">
            <label>Cost Value</label>
            <input
              type="number"
              placeholder="Enter cost"
              value={cost}
              onChange={(e) => {
                setCost(e.target.value);
                setError('');
              }}
              className={error ? 'input-error' : ''}
              min="0"
            />
          </div>
          {error && <div className="error-message">{error}</div>}
        </div>
        <div className="modal-footer">
          <button className="modal-cancel-btn" onClick={onClose}>Cancel</button>
          <button className="modal-add-btn" onClick={handleAdd}>Add Cost</button>
        </div>
      </div>
    </div>
  );
};

export default EdgeCostModal; 