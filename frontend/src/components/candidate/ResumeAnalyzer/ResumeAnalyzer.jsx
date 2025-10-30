// frontend/src/components/candidate/ResumeAnalyzer/ResumeAnalyzer.jsx
import { useState } from 'react';
import FileUpload from './FileUpload';
import SkillSelectionModal from './SkillSelectionModal';
import DiffViewer from './DiffViewer';
import { useResumeAnalysis } from '../../../hooks/useResumeAnalysis';

export default function ResumeAnalyzer() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jdFile, setJdFile] = useState(null);

  const {
    analyzeResume,
    generateOptimized,
    generatePDF,
    toggleSkill,
    selectAllSkills,
    deselectAllSkills,
    handleCancelSkillSelection,
    currentStep,
    missingSkills,
    selectedSkills,
    improvementTips,
    optimizedResume,
    originalResume, // ✅ ADD THIS
    loading,
    error,
    uploadStatus,
    selectedSkillsCount
  } = useResumeAnalysis();

  const handleAnalyze = async () => {
    if (!resumeFile || !jdFile) {
      alert('Please upload both files');
      return;
    }
    await analyzeResume(resumeFile, jdFile);
  };

  const handleContinueWithSkills = async () => {
    await generateOptimized(resumeFile, jdFile);
  };

  const handleDownload = () => {
    if (!optimizedResume) return;
    
    const blob = new Blob([optimizedResume], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'optimized_resume.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  const handleTestConnection = async () => {
    try {
      const response = await fetch(import.meta.env.VITE_RESUME_ANALYZER_URL + '/');
      if (response.ok) {
        alert('✅ Connection successful! AI service is running.');
      } else {
        alert('⚠️ AI service responded but with an error.');
      }
    } catch (err) {
      alert('❌ Cannot connect to AI service. Make sure the Resume Analyzer service is running.');
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h2 style={styles.title}>AI Resume Analyzer & Optimizer</h2>
        <p style={styles.subtitle}>
          Upload your resume and a job description to get AI-powered optimization suggestions
        </p>

        {/* File Upload Section */}
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>Upload Files</h3>
          <div style={styles.uploadGrid}>
            <FileUpload
              id="resume-upload"
              label="Resume (PDF)"
              icon="📄"
              accept=".pdf"
              file={resumeFile}
              onFileSelect={setResumeFile}
              description="Click to upload your resume"
            />
            <FileUpload
              id="jd-upload"
              label="Job Description (PDF)"
              icon="💼"
              accept=".pdf"
              file={jdFile}
              onFileSelect={setJdFile}
              description="Click to upload job description"
            />
          </div>
          <p style={styles.requirements}>
            <strong>Requirements:</strong> PDF files only, maximum 10MB each
          </p>
        </div>

        {/* Action Buttons */}
        <div style={styles.section}>
          <div style={styles.buttonContainer}>
            <button
              onClick={handleTestConnection}
              style={{
                ...styles.button,
                ...styles.secondaryButton,
                cursor: 'pointer'
              }}
            >
              🔌 Test Connection
            </button>

            <button
              onClick={handleAnalyze}
              disabled={!resumeFile || !jdFile || loading}
              style={{
                ...styles.button,
                ...styles.primaryButton,
                opacity: (!resumeFile || !jdFile || loading) ? 0.6 : 1,
                cursor: (!resumeFile || !jdFile || loading) ? 'not-allowed' : 'pointer'
              }}
            >
              🔍 Analyze Resume
            </button>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div style={styles.loadingContainer}>
            <div style={styles.spinner}></div>
            <p style={styles.loadingText}>
              {uploadStatus || 'Analyzing your resume...'}
            </p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div style={styles.errorContainer}>
            <p style={styles.errorText}>{error}</p>
          </div>
        )}

        {/* Results Section */}
        {currentStep && !loading && (
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Analysis Results</h3>
            
            {/* Missing Skills */}
            {missingSkills && missingSkills.length > 0 && (
              <div style={styles.resultItem}>
                <h4 style={styles.resultTitle}>Missing Skills Found ({missingSkills.length})</h4>
                <div style={styles.skillsList}>
                  {missingSkills.map((skill, index) => (
                    <div key={index} style={styles.skillItem}>
                      {skill}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Improvement Tips */}
            {improvementTips && improvementTips.length > 0 && (
              <div style={styles.resultItem}>
                <h4 style={styles.resultTitle}>Improvement Suggestions</h4>
                <ul style={styles.tipsList}>
                  {improvementTips.map((tip, index) => (
                    <li key={index} style={styles.tipItem}>{tip}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Action Buttons for Results */}
            <div style={styles.buttonContainer}>
              {missingSkills && missingSkills.length > 0 && (
                <button
                  onClick={handleContinueWithSkills}
                  disabled={loading}
                  style={{
                    ...styles.button,
                    ...styles.primaryButton,
                    opacity: loading ? 0.6 : 1,
                    cursor: loading ? 'not-allowed' : 'pointer'
                  }}
                >
                  Continue with Selected Skills ({selectedSkillsCount})
                </button>
              )}
              
              {optimizedResume && (
                <button
                  onClick={handleDownload}
                  style={{
                    ...styles.button,
                    ...styles.secondaryButton,
                    cursor: 'pointer'
                  }}
                >
                  📄 Download Optimized Resume
                </button>
              )}
            </div>
          </div>
        )}

        {/* Instructions */}
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>How it works:</h3>
          <ol style={styles.instructions}>
            <li>Upload your current resume (PDF)</li>
            <li>Upload the job description you're targeting (PDF)</li>
            <li><strong>Test Connection</strong> (optional) - Verify backend connectivity</li>
            <li><strong>Analyze Resume</strong> - AI identifies missing skills</li>
            <li>Select which skills you want to add</li>
            <li>Get your optimized resume instantly!</li>
          </ol>
        </div>
      </div>

      {/* Skill Selection Modal */}
      <SkillSelectionModal
        isOpen={currentStep === 'skillSelection'}
        missingSkills={missingSkills}
        selectedSkills={selectedSkills}
        onToggleSkill={toggleSkill}
        onSelectAll={selectAllSkills}
        onDeselectAll={deselectAllSkills}
        onContinue={handleContinueWithSkills}
        onCancel={handleCancelSkillSelection}
      />

      {/* Diff Viewer Modal */}
      <DiffViewer
        isOpen={currentStep === 'diffView'}
        originalText={originalResume}
        optimizedText={optimizedResume}
        onDownload={handleDownload}
      />
    </div>
  );
}

const styles = {
  container: {
    padding: '20px',
    maxWidth: '1200px',
    margin: '0 auto',
    fontFamily: 'Arial, sans-serif'
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: '12px',
    padding: '30px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    border: '1px solid #e5e7eb'
  },
  title: {
    fontSize: '28px',
    fontWeight: 'bold',
    color: '#1f2937',
    textAlign: 'center',
    marginBottom: '10px'
  },
  subtitle: {
    fontSize: '16px',
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: '30px'
  },
  section: {
    marginBottom: '30px'
  },
  sectionTitle: {
    fontSize: '20px',
    fontWeight: '600',
    color: '#374151',
    marginBottom: '15px'
  },
  uploadGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '20px',
    marginBottom: '15px'
  },
  requirements: {
    fontSize: '14px',
    color: '#6b7280',
    margin: '0'
  },
  buttonContainer: {
    display: 'flex',
    gap: '15px',
    justifyContent: 'center'
  },
  button: {
    padding: '12px 24px',
    border: 'none',
    borderRadius: '8px',
    fontSize: '16px',
    fontWeight: '500',
    transition: 'all 0.2s ease'
  },
  primaryButton: {
    backgroundColor: '#3b82f6',
    color: 'white'
  },
  secondaryButton: {
    backgroundColor: '#f3f4f6',
    color: '#374151',
    border: '1px solid #d1d5db'
  },
  loadingContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '40px'
  },
  spinner: {
    width: '40px',
    height: '40px',
    border: '4px solid #f3f4f6',
    borderTop: '4px solid #3b82f6',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
    marginBottom: '15px'
  },
  loadingText: {
    fontSize: '16px',
    color: '#6b7280',
    margin: '0'
  },
  errorContainer: {
    backgroundColor: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: '8px',
    padding: '15px',
    marginBottom: '20px'
  },
  errorText: {
    color: '#dc2626',
    margin: '0',
    fontSize: '14px'
  },
  resultItem: {
    marginBottom: '25px'
  },
  resultTitle: {
    fontSize: '18px',
    fontWeight: '600',
    color: '#374151',
    marginBottom: '10px'
  },
  skillsList: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: '8px'
  },
  skillItem: {
    backgroundColor: '#fef3c7',
    border: '1px solid #fcd34d',
    borderRadius: '6px',
    padding: '8px 12px',
    fontSize: '14px',
    color: '#92400e'
  },
  tipsList: {
    margin: '0',
    paddingLeft: '20px'
  },
  tipItem: {
    marginBottom: '8px',
    color: '#4b5563',
    lineHeight: '1.5'
  },
  instructions: {
    margin: '0',
    paddingLeft: '20px',
    color: '#4b5563'
  },
  instructions: {
    margin: '0',
    paddingLeft: '20px'
  },
  instructions: {
    margin: '0',
    paddingLeft: '20px'
  },
  instructions: {
    margin: '0',
    paddingLeft: '20px'
  }
};