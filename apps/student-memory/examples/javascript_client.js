/**
 * JavaScript / Node.js client for Student Personalization Memory Service
 */

class MemoryClient {
  constructor(baseUrl = 'http://localhost:8000', serviceKey = 'development-secret-key') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.serviceKey = serviceKey;
  }

  async _fetch(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      'X-Service-Key': this.serviceKey,
      ...options.headers,
    };

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(`Memory Service Error ${response.status}: ${errorBody}`);
    }

    return response.json();
  }

  /**
   * Classify a student question into a single canonical topic.
   * @param {string} question 
   * @returns {Promise<{topic: string|null, skill_id: string|null, confidence: number, is_math: boolean}>}
   */
  async classifyTopic(question) {
    return this._fetch('/topic/classify', {
      method: 'POST',
      body: JSON.stringify({ question }),
    });
  }

  /**
   * Update student memory with Evaluator assessment outcomes.
   */
  async updateMemory({ studentId, topic, assessmentQuestions, identifiedErrors = [], overallFeedback = null }) {
    return this._fetch('/memory/update', {
      method: 'POST',
      body: JSON.stringify({
        student_id: studentId,
        topic,
        assessment_questions: assessmentQuestions,
        identified_errors: identifiedErrors,
        overall_feedback: overallFeedback,
      }),
    });
  }

  /**
   * Retrieve Tutor personalization context for a student.
   */
  async getTutorContext(studentId) {
    return this._fetch(`/memory/${encodeURIComponent(studentId)}/tutor-context`);
  }

  /**
   * Retrieve Planner personalization context for a student.
   */
  async getPlannerContext(studentId) {
    return this._fetch(`/memory/${encodeURIComponent(studentId)}/planner-context`);
  }
}

module.exports = { MemoryClient };
