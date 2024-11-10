import React, { useEffect } from 'react';
import './GlobalNotification.css';

const GlobalNotification = ({ message, type = 'info', onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 3000);  

    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={`global-notification-slide notification-${type}`}>
      <div className="global-notification-content">
        <button className="notification-close-btn" onClick={onClose}>×</button>
        <div className="notification-message">{message}</div>
      </div>
    </div>
  );
};

export default GlobalNotification; 