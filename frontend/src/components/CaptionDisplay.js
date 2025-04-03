import React from 'react';

const CaptionDisplay = ({ caption, loading, error }) => {
  return (
    <div className="caption-section">
      <h2>Generated Caption</h2>
      <div className="caption-container">
        {loading ? (
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>Generating caption...</p>
          </div>
        ) : error ? (
          <div className="error-message">
            <p>{error}</p>
          </div>
        ) : caption ? (
          <div className="caption-result">
            <p className="caption-text">{caption}</p>
          </div>
        ) : (
          <div className="no-caption">
            <p>Upload an image to generate a caption</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default CaptionDisplay;