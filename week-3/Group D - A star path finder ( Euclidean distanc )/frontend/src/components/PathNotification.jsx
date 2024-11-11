import React, { useEffect } from 'react';
import './PathNotification.css';

const PathNotification = ({ pathInfo, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 5000);  

    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className="path-notification-slide">
      <div className="path-notification-content">
        <button className="notification-close-btn" onClick={onClose}>×</button>
        <h3>Path Found!</h3>
        <div className="path-info">
          <div className="path-row">
            <strong>Path:</strong>
            <span>{pathInfo.pathNames}</span>
          </div>
          <div className="path-row">
            <strong>Cost:</strong>
            <span>{pathInfo.totalCost}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PathNotification; 