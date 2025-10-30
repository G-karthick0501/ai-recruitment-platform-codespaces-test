// frontend/src/components/candidate/ResumeAnalyzer/DiffViewer.jsx

import React, { useMemo } from 'react';

export default function DiffViewer({ originalText, optimizedText, addedSkills }) {
  // ✅ ADD SAFETY CHECKS
  if (!originalText || !optimizedText) {
    return null;
  }

  // Calculate statistics
  const stats = useMemo(() => {
    // ✅ ADD SAFETY CHECKS
    if (!originalText || !optimizedText) {
      return {
        originalLines: 0,
        optimizedLines: 0,
        addedLines: 0,
        addedSkillsCount: 0
      };
    }
    
    const originalLines = originalText.split('\n').filter(l => l.trim()).length;
    const optimizedLines = optimizedText.split('\n').filter(l => l.trim()).length;
    const addedLines = optimizedLines - originalLines;
    
    return {
      originalLines,
      optimizedLines,
      addedLines,
      addedSkillsCount: addedSkills ? addedSkills.length : 0
    };
  }, [originalText, optimizedText, addedSkills]);

  // Render optimized text with skill highlights
  const renderHighlightedText = (text, skills) => {
    // ✅ ADD SAFETY CHECKS
    if (!text || !skills) {
      return text || '';
    }
    
    let highlightedText = text;
    const highlightPositions = [];
    
    // Find positions of each skill
    skills.forEach(skill => {
      if (!skill) return;
      
      const regex = new RegExp(`(${escapeRegex(skill)})`, 'gi');
      let match;
      while ((match = regex.exec(text)) !== null) {
        highlightPositions.push({
          start: match.index,
          end: match.index + match[0].length,
          skill: match[0]
        });
      }
    });
    
    // Sort by position
    highlightPositions.sort((a, b) => a.start - b.start);
    
    // Build highlighted text
    let result = [];
    let lastIndex = 0;
    
    highlightPositions.forEach((pos, idx) => {
      // Add text before highlight
      if (pos.start > lastIndex) {
        result.push(
          <span key={`text-${idx}`}>
            {text.substring(lastIndex, pos.start)}
          </span>
        );
      }
      
      // Add highlighted skill
      result.push(
        <span key={`highlight-${idx}`} style={{
          backgroundColor: '#fef3c7',
          padding: '2px 4px',
          borderRadius: '3px',
          fontWeight: 'bold',
          color: '#92400e'
        }}>
          {text.substring(pos.start, pos.end)}
        </span>
      );
      
      lastIndex = pos.end;
    });
    
    // Add remaining text
    if (lastIndex < text.length) {
      result.push(
        <span key="text-final">
          {text.substring(lastIndex)}
        </span>
      );
    }
    
    return result;
  };

  // Helper function to escape regex special characters
  const escapeRegex = (string) => {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  };

  return (
    <div style={{
      border: '1px solid #e5e7eb',
      borderRadius: '8px',
      padding: '20px',
      backgroundColor: '#f9fafb',
      marginBottom: '20px'
    }}>
      <h4 style={{
        fontSize: '18px',
        fontWeight: '600',
        color: '#374151',
        marginBottom: '15px'
      }}>
        📊 Resume Optimization Results
      </h4>
      
      {/* Statistics */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '15px',
        marginBottom: '20px',
        padding: '15px',
        backgroundColor: '#ffffff',
        borderRadius: '6px',
        border: '1px solid #e5e7eb'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#3b82f6' }}>
            {stats.originalLines}
          </div>
          <div style={{ fontSize: '14px', color: '#6b7280' }}>Original Lines</div>
        </div>
        
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#10b981' }}>
            {stats.optimizedLines}
          </div>
          <div style={{ fontSize: '14px', color: '#6b7280' }}>Optimized Lines</div>
        </div>
        
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#f59e0b' }}>
            {stats.addedLines > 0 ? `+${stats.addedLines}` : stats.addedLines}
          </div>
          <div style={{ fontSize: '14px', color: '#6b7280' }}>Lines Added</div>
        </div>
        
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#8b5cf6' }}>
            {stats.addedSkillsCount}
          </div>
          <div style={{ fontSize: '14px', color: '#6b7280' }}>Skills Added</div>
        </div>
      </div>
      
      {/* Optimized Resume Preview */}
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '6px',
        padding: '15px',
        border: '1px solid #e5e7eb'
      }}>
        <h5 style={{
          fontSize: '16px',
          fontWeight: '600',
          color: '#374151',
          marginBottom: '10px'
        }}>
          📄 Optimized Resume Preview
        </h5>
        
        <div style={{
          fontFamily: 'monospace',
          fontSize: '14px',
          lineHeight: '1.5',
          whiteSpace: 'pre-wrap',
          maxHeight: '400px',
          overflowY: 'auto',
          padding: '10px',
          backgroundColor: '#f9fafb',
          borderRadius: '4px',
          border: '1px solid #e5e7eb'
        }}>
          {renderHighlightedText(optimizedText, addedSkills || [])}
        </div>
      </div>
      
      {/* Legend */}
      <div style={{
        marginTop: '15px',
        padding: '10px',
        backgroundColor: '#ffffff',
        borderRadius: '6px',
        border: '1px solid #e5e7eb',
        fontSize: '12px',
        color: '#6b7280'
      }}>
        <strong>Legend:</strong> 
        <span style={{
          backgroundColor: '#fef3c7',
          padding: '2px 4px',
          borderRadius: '3px',
          marginLeft: '10px'
        }}>
          Highlighted skills were added based on job description requirements
        </span>
      </div>
    </div>
  );
}