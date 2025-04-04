import React, { useState } from 'react';
import './styles.css';
import ImageUploader from './components/ImageUploader';
import CaptionDisplay from './components/CaptionDisplay';
import Header from './components/Header';

function App() {
  const [image, setImage] = useState(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImageUpload = async (imageFile) => {
    setImage(imageFile);
    setLoading(true);
    setError('');
    setCaption('');
    
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      const response = await fetch('http://localhost:8000/api/caption', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setCaption(data.caption);
    } catch (err) {
      setError('Failed to generate caption. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Header />
      <main className="main-content">
        <div className="container">
          <ImageUploader onImageUpload={handleImageUpload} />
          
          {image && (
            <div className="preview-container">
              <h3>Image Preview</h3>
              <img 
                src={URL.createObjectURL(image)} 
                alt="Preview" 
                className="image-preview" 
              />
            </div>
          )}
          
          <CaptionDisplay 
            caption={caption} 
            loading={loading} 
            error={error} 
          />
        </div>
      </main>
      <footer className="footer">
        <p>Image Caption Generator &copy; {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;