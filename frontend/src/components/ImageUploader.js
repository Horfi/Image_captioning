import React, { useRef, useState } from 'react';

const ImageUploader = ({ onImageUpload }) => {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (isValidImageFile(file)) {
        onImageUpload(file);
      } else {
        alert('Please upload a valid image file (jpg, jpeg, png, gif)');
      }
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (isValidImageFile(file)) {
        onImageUpload(file);
      } else {
        alert('Please upload a valid image file (jpg, jpeg, png, gif)');
      }
    }
  };

  const isValidImageFile = (file) => {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    return validTypes.includes(file.type);
  };

  const onButtonClick = () => {
    inputRef.current.click();
  };

  return (
    <div className="uploader-section">
      <h2>Upload an Image</h2>
      <div 
        className={`drag-drop-area ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          className="input-file"
          accept="image/*"
          onChange={handleChange}
        />
        <div className="drag-drop-content">
          <p>Drag and drop an image here or</p>
          <button className="upload-button" onClick={onButtonClick}>
            Select File
          </button>
          <p className="file-types">Supports: JPG, JPEG, PNG, GIF</p>
        </div>
      </div>
    </div>
  );
};

export default ImageUploader;