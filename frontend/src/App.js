import React, { useState, useEffect } from 'react';
import './styles.css';
import ImageUploader from './components/ImageUploader';
import CaptionDisplay from './components/CaptionDisplay';
import Header from './components/Header';
import { initModel, predictCaption } from './utils/model';

function App() {
  const [image, setImage] = useState(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelReady, setModelReady] = useState(false);

   // 1) Load model & vocab once
   useEffect(() => {
    initModel()
      .then(() => setModelReady(true))
      .catch(err => {
        console.error('TF.js load failed:', err);
        setError('Could not load ML model in browser');
      });
  }, []);

  const handleImageUpload = async (imageFile) => {
    if (!modelReady) {
      setError('Model still loading… please wait');
      return;
    }
    setImage(imageFile);
    setLoading(true);
    setError('');
    setCaption('');
    
    try {
    // run in-browser TF.js inference
    const cap = await predictCaption(imageFile);
    setCaption(cap);
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