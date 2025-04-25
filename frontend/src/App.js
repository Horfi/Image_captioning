import React, { useState } from 'react';
import './styles.css';
import ImageUploader from './components/ImageUploader';
import CaptionDisplay from './components/CaptionDisplay';
import Header from './components/Header';
import { fetchCaption } from './utils/api';

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImageUpload = async (file) => {
    setImageFile(file);
    setLoading(true);
    setError('');
    setCaption('');

    try {
      const { caption: text } = await fetchCaption(file);
      setCaption(text);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to generate caption');
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

          {imageFile && (
            <div className="preview-container">
              <h3>Image Preview</h3>
              <img
                src={URL.createObjectURL(imageFile)}
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
        <p>Image Caption Generator © {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;