import React, { useState } from 'react';
import './App.css';
import { useTranslation } from 'react-i18next';
import './i18n/i18n'; // Import the i18n config

const nutrientMap = { N: "Nitrogen", P: "Phosphorus", K: "Potassium" };

const App = () => {
  const { t, i18n } = useTranslation();

  const [formData, setFormData] = useState({
    Nitrogen: '',
    Phosphorus: '',
    Potassium: '',
    Temperature: '',
    Humidity: '',
    ph: '',
    Rainfall: ''
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResult(null);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          N: parseFloat(formData.Nitrogen),
          P: parseFloat(formData.Phosphorus),
          K: parseFloat(formData.Potassium),
          temperature: parseFloat(formData.Temperature),
          humidity: parseFloat(formData.Humidity),
          ph: parseFloat(formData.ph),
          rainfall: parseFloat(formData.Rainfall)
        })
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (e) {
      console.error("Failed to fetch:", e);
      setError(t('Failed'));
    } finally {
      setIsLoading(false);
    }
  };

  // Language toggle
  const changeLanguage = (lang) => i18n.changeLanguage(lang);

  return (
    <div className="app-container">
      <div className="card">
        <h1 className="title">{t('title')}</h1>
        <p className="subtitle">{t('subtitle')}</p>

        {/* Language buttons */}
        <div style={{ marginBottom: '10px' }}>
          <button onClick={() => changeLanguage('en')}>English</button>
          <button onClick={() => changeLanguage('hi')}>हिन्दी</button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            {Object.keys(formData).map(key => (
              <div key={key} className="form-group">
                <label htmlFor={key} className="label">{t(key)}</label>
                <input
                  type="number"
                  id={key}
                  name={key}
                  value={formData[key]}
                  onChange={handleChange}
                  required
                  className="input-field"
                />
              </div>
            ))}
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className={`submit-button ${isLoading ? 'loading' : ''}`}
          >
            {isLoading ? t('Predicting') : t('GetRecommendation')}
          </button>
        </form>

        {result && (
  <div className="result-box success">
    <h2 className="result-title">{t('Recommendation')}</h2>
    
    <p className="result-item">
      {t('Crop')}: <span className="result-text">{t(result.recommended_crop)}</span>
    </p>

    {['N','P','K'].map(key =>
      result[key] !== undefined && (
        <p key={key} className="result-item">
          {t(nutrientMap[key])}: <span className="result-text">{result[key]}</span>
        </p>
      )
    )}

    {result.risk_factors && (
      <p className="result-item">
        {t('RiskLevel')}: <span className="result-text">{t(result.risk_factors)}</span>
      </p>
    )}

    {result.annual_income && (
      <p className="result-item">
        {t('EstimatedIncome')}: <span className="result-text">{result.annual_income}</span>
      </p>
    )}
  </div>
)}

        {error && (
          <div className="result-box error">
            <p className="result-text">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
