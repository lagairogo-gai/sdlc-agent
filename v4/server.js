// server.js - Main Express server
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { OpenAI } = require('openai');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const pdf = require('pdf-parse');
const mammoth = require('mammoth');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// File upload configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});
const upload = multer({ storage });

// LLM Clients
const clients = {
  openai: null,
  azure: null,
  gemini: null,
  claude: null
};

// Initialize LLM clients
const initializeClients = () => {
  if (process.env.OPENAI_API_KEY) {
    clients.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }
  
  if (process.env.AZURE_OPENAI_ENDPOINT && process.env.AZURE_OPENAI_API_KEY) {
    clients.azure = new OpenAI({
      apiKey: process.env.AZURE_OPENAI_API_KEY,
      baseURL: `${process.env.AZURE_OPENAI_ENDPOINT}/openai/deployments/${process.env.AZURE_OPENAI_DEPLOYMENT}`,
      defaultQuery: { 'api-version': '2024-02-15-preview' },
      defaultHeaders: {
        'api-key': process.env.AZURE_OPENAI_API_KEY,
      },
    });
  }
};

initializeClients();

// Data source handlers
class DataSourceManager {
  constructor() {
    this.jiraClient = null;
    this.confluenceClient = null;
    this.sharepointClient = null;
  }

  // Jira Integration
  async connectJira(credentials) {
    try {
      const { url, username, apiToken } = credentials;
      const auth = Buffer.from(`${username}:${apiToken}`).toString('base64');
      
      const response = await axios.get(`${url}/rest/api/2/myself`, {
        headers: {
          'Authorization': `Basic ${auth}`,
          'Accept': 'application/json'
        }
      });
      
      this.jiraClient = { url, auth };
      return { success: true, user: response.data };
    } catch (error) {
      console.error('Jira connection failed:', error.message);
      throw new Error('Failed to connect to Jira');
    }
  }

  async fetchJiraRequirements(projectKey) {
    if (!this.jiraClient) throw new Error('Jira not connected');
    
    try {
      const response = await axios.get(
        `${this.jiraClient.url}/rest/api/2/search?jql=project=${projectKey} AND type in (Epic,Story,Requirement)`,
        {
          headers: {
            'Authorization': `Basic ${this.jiraClient.auth}`,
            'Accept': 'application/json'
          }
        }
      );
      
      return response.data.issues.map(issue => ({
        id: issue.key,
        summary: issue.fields.summary,
        description: issue.fields.description || '',
        issueType: issue.fields.issuetype.name,
        priority: issue.fields.priority?.name || 'Medium',
        components: issue.fields.components.map(c => c.name),
        labels: issue.fields.labels
      }));
    } catch (error) {
      console.error('Failed to fetch Jira requirements:', error.message);
      throw new Error('Failed to fetch Jira requirements');
    }
  }

  // Confluence Integration
  async connectConfluence(credentials) {
    try {
      const { url, username, apiToken } = credentials;
      const auth = Buffer.from(`${username}:${apiToken}`).toString('base64');
      
      const response = await axios.get(`${url}/rest/api/user/current`, {
        headers: {
          'Authorization': `Basic ${auth}`,
          'Accept': 'application/json'
        }
      });
      
      this.confluenceClient = { url, auth };
      return { success: true, user: response.data };
    } catch (error) {
      console.error('Confluence connection failed:', error.message);
      throw new Error('Failed to connect to Confluence');
    }
  }

  async fetchConfluencePages(spaceKey) {
    if (!this.confluenceClient) throw new Error('Confluence not connected');
    
    try {
      const response = await axios.get(
        `${this.confluenceClient.url}/rest/api/content?spaceKey=${spaceKey}&type=page&expand=body.storage`,
        {
          headers: {
            'Authorization': `Basic ${this.confluenceClient.auth}`,
            'Accept': 'application/json'
          }
        }
      );
      
      return response.data.results.map(page => ({
        id: page.id,
        title: page.title,
        content: page.body.storage.value.replace(/<[^>]*>/g, ''), // Strip HTML
        lastModified: page.version.when,
        space: page.space.name
      }));
    } catch (error) {
      console.error('Failed to fetch Confluence pages:', error.message);
      throw new Error('Failed to fetch Confluence pages');
    }
  }

  // SharePoint Integration (using Microsoft Graph API)
  async connectSharePoint(credentials) {
    try {
      const { tenantId, clientId, clientSecret, siteUrl } = credentials;
      
      // Get access token
      const tokenResponse = await axios.post(
        `https://login.microsoftonline.com/${tenantId}/oauth2/v2.0/token`,
        new URLSearchParams({
          client_id: clientId,
          client_secret: clientSecret,
          scope: 'https://graph.microsoft.com/.default',
          grant_type: 'client_credentials'
        }),
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
          }
        }
      );
      
      this.sharepointClient = {
        accessToken: tokenResponse.data.access_token,
        siteUrl
      };
      
      return { success: true };
    } catch (error) {
      console.error('SharePoint connection failed:', error.message);
      throw new Error('Failed to connect to SharePoint');
    }
  }

  async fetchSharePointDocuments(driveId, folderId = 'root') {
    if (!this.sharepointClient) throw new Error('SharePoint not connected');
    
    try {
      const response = await axios.get(
        `https://graph.microsoft.com/v1.0/drives/${driveId}/items/${folderId}/children`,
        {
          headers: {
            'Authorization': `Bearer ${this.sharepointClient.accessToken}`,
            'Accept': 'application/json'
          }
        }
      );
      
      const documents = [];
      for (const item of response.data.value) {
        if (item.file && (item.name.endsWith('.docx') || item.name.endsWith('.pdf'))) {
          // Download and extract content
          const contentResponse = await axios.get(item['@microsoft.graph.downloadUrl'], {
            responseType: 'arraybuffer'
          });
          
          let content = '';
          if (item.name.endsWith('.docx')) {
            const result = await mammoth.extractRawText({ buffer: contentResponse.data });
            content = result.value;
          } else if (item.name.endsWith('.pdf')) {
            const pdfData = await pdf(contentResponse.data);
            content = pdfData.text;
          }
          
          documents.push({
            id: item.id,
            name: item.name,
            content,
            lastModified: item.lastModifiedDateTime,
            size: item.size
          });
        }
      }
      
      return documents;
    } catch (error) {
      console.error('Failed to fetch SharePoint documents:', error.message);
      throw new Error('Failed to fetch SharePoint documents');
    }
  }

  // Export user stories to Jira
  async exportToJira(userStories, projectKey, issueTypeId = '7') {
    if (!this.jiraClient) throw new Error('Jira not connected');
    
    try {
      const createdIssues = [];
      
      for (const story of userStories) {
        const issueData = {
          fields: {
            project: { key: projectKey },
            summary: story.title,
            description: this.formatDescriptionForJira(story),
            issuetype: { id: issueTypeId }, // Story issue type
            priority: { name: story.priority || 'Medium' }
          }
        };
        
        if (story.storyPoints) {
          issueData.fields.customfield_10016 = story.storyPoints; // Story Points field
        }
        
        const response = await axios.post(
          `${this.jiraClient.url}/rest/api/2/issue`,
          issueData,
          {
            headers: {
              'Authorization': `Basic ${this.jiraClient.auth}`,
              'Content-Type': 'application/json'
            }
          }
        );
        
        createdIssues.push({
          key: response.data.key,
          id: response.data.id,
          title: story.title
        });
      }
      
      return createdIssues;
    } catch (error) {
      console.error('Failed to export to Jira:', error.message);
      throw new Error('Failed to export user stories to Jira');
    }
  }

  formatDescriptionForJira(story) {
    let description = story.description + '\n\n';
    
    if (story.acceptanceCriteria && story.acceptanceCriteria.length > 0) {
      description += '*Acceptance Criteria:*\n';
      story.acceptanceCriteria.forEach(criteria => {
        description += `* ${criteria}\n`;
      });
    }
    
    return description;
  }
}

// Document processing utilities
class DocumentProcessor {
  static async extractTextFromFile(filePath, mimeType) {
    try {
      if (mimeType.includes('pdf')) {
        const dataBuffer = await fs.readFile(filePath);
        const pdfData = await pdf(dataBuffer);
        return pdfData.text;
      } else if (mimeType.includes('word') || filePath.endsWith('.docx')) {
        const result = await mammoth.extractRawText({ path: filePath });
        return result.value;
      } else if (mimeType.includes('text') || filePath.endsWith('.txt')) {
        return await fs.readFile(filePath, 'utf-8');
      }
      
      throw new Error('Unsupported file type');
    } catch (error) {
      console.error('Error extracting text from file:', error);
      throw error;
    }
  }

  static chunkText(text, maxChunkSize = 1000, overlap = 200) {
    const chunks = [];
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    let currentChunk = '';
    let currentSize = 0;
    
    for (const sentence of sentences) {
      const sentenceSize = sentence.length;
      
      if (currentSize + sentenceSize > maxChunkSize && currentChunk) {
        chunks.push(currentChunk.trim());
        
        // Create overlap
        const words = currentChunk.split(' ');
        const overlapWords = words.slice(-Math.floor(overlap / 10));
        currentChunk = overlapWords.join(' ') + ' ';
        currentSize = currentChunk.length;
      }
      
      currentChunk += sentence + '. ';
      currentSize += sentenceSize + 2;
    }
    
    if (currentChunk.trim()) {
      chunks.push(currentChunk.trim());
    }
    
    return chunks;
  }
}

// Vector store for RAG
class VectorStore {
  constructor() {
    this.documents = [];
    this.embeddings = [];
  }

  async addDocuments(documents) {
    for (const doc of documents) {
      const chunks = DocumentProcessor.chunkText(doc.content);
      
      for (let i = 0; i < chunks.length; i++) {
        this.documents.push({
          id: `${doc.id}_chunk_${i}`,
          content: chunks[i],
          metadata: {
            source: doc.source || 'upload',
            title: doc.title || doc.name,
            chunkIndex: i,
            totalChunks: chunks.length
          }
        });
      }
    }
  }

  async generateEmbeddings(client) {
    if (!client) throw new Error('No LLM client available for embeddings');
    
    try {
      const texts = this.documents.map(doc => doc.content);
      const response = await client.embeddings.create({
        model: 'text-embedding-ada-002',
        input: texts,
      });
      
      this.embeddings = response.data.map(item => item.embedding);
    } catch (error) {
      console.error('Error generating embeddings:', error);
      throw error;
    }
  }

  cosineSimilarity(a, b) {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (normA * normB);
  }

  async search(query, client, topK = 5) {
    if (!client) throw new Error('No LLM client available for search');
    
    try {
      const queryResponse = await client.embeddings.create({
        model: 'text-embedding-ada-002',
        input: query,
      });
      
      const queryEmbedding = queryResponse.data[0].embedding;
      
      const similarities = this.embeddings.map((embedding, index) => ({
        index,
        similarity: this.cosineSimilarity(queryEmbedding, embedding),
        document: this.documents[index]
      }));
      
      return similarities
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, topK)
        .map(item => item.document);
    } catch (error) {
      console.error('Error searching vector store:', error);
      throw error;
    }
  }
}

// User story generator
class UserStoryGenerator {
  constructor() {
    this.vectorStore = new VectorStore();
  }

  async generateUserStories(requirements, llmConfig) {
    const client = clients[llmConfig.provider];
    if (!client) throw new Error(`LLM provider ${llmConfig.provider} not configured`);

    try {
      // Search for relevant context
      const relevantDocs = await this.vectorStore.search(
        requirements.join(' '), 
        client, 
        10
      );
      
      const context = relevantDocs.map(doc => doc.content).join('\n\n');
      
      const prompt = this.buildPrompt(requirements, context);
      
      const response = await client.chat.completions.create({
        model: llmConfig.model,
        messages: [
          {
            role: 'system',
            content: 'You are an expert product manager and business analyst. Generate well-structured user stories from requirements.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: llmConfig.temperature,
        max_tokens: 4000
      });

      return this.parseUserStories(response.choices[0].message.content);
    } catch (error) {
      console.error('Error generating user stories:', error);
      throw error;
    }
  }

  buildPrompt(requirements, context) {
    return `
Based on the following requirements and context, generate comprehensive user stories in JSON format.

REQUIREMENTS:
${requirements.join('\n')}

CONTEXT FROM DOCUMENTS:
${context}

Please generate user stories following this exact JSON structure:
{
  "userStories": [
    {
      "id": "unique_id",
      "title": "Story Title",
      "description": "As a [user type], I want [goal] so that [reason/benefit]",
      "acceptanceCriteria": [
        "Criteria 1",
        "Criteria 2",
        "Criteria 3"
      ],
      "priority": "High|Medium|Low",
      "storyPoints": number,
      "epic": "Epic name if applicable",
      "labels": ["label1", "label2"],
      "components": ["component1", "component2"]
    }
  ]
}

Guidelines:
1. Follow the "As a... I want... So that..." format
2. Include 3-5 specific, testable acceptance criteria
3. Assign appropriate story points (1, 2, 3, 5, 8, 13)
4. Set realistic priorities based on business value
5. Group related stories under epics
6. Add relevant labels and components
7. Ensure stories are independent and deliverable
8. Focus on user value and business outcomes

Generate 5-10 comprehensive user stories.`;
  }

  parseUserStories(content) {
    try {
      // Extract JSON from the response
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in response');
      }
      
      const parsed = JSON.parse(jsonMatch[0]);
      return parsed.userStories || [];
    } catch (error) {
      console.error('Error parsing user stories:', error);
      // Fallback: try to extract stories manually
      return this.fallbackParse(content);
    }
  }

  fallbackParse(content) {
    // Simple fallback parser for malformed JSON responses
    const stories = [];
    const lines = content.split('\n');
    
    let currentStory = null;
    
    for (const line of lines) {
      if (line.includes('As a') || line.includes('As an')) {
        if (currentStory) {
          stories.push(currentStory);
        }
        currentStory = {
          id: `story_${stories.length + 1}`,
          title: line.trim(),
          description: line.trim(),
          acceptanceCriteria: [],
          priority: 'Medium',
          storyPoints: 5
        };
      } else if (currentStory && line.trim().startsWith('-')) {
        currentStory.acceptanceCriteria.push(line.trim().substring(1).trim());
      }
    }
    
    if (currentStory) {
      stories.push(currentStory);
    }
    
    return stories;
  }
}

// Initialize data source manager and user story generator
const dataSourceManager = new DataSourceManager();
const userStoryGenerator = new UserStoryGenerator();

// Create uploads directory
const createUploadsDir = async () => {
  try {
    await fs.mkdir('uploads', { recursive: true });
  } catch (error) {
    console.log('Uploads directory already exists');
  }
};

createUploadsDir();

// API Routes

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Data source connections
app.post('/api/connect/jira', async (req, res) => {
  try {
    const result = await dataSourceManager.connectJira(req.body);
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.post('/api/connect/confluence', async (req, res) => {
  try {
    const result = await dataSourceManager.connectConfluence(req.body);
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.post('/api/connect/sharepoint', async (req, res) => {
  try {
    const result = await dataSourceManager.connectSharePoint(req.body);
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Fetch data from sources
app.get('/api/jira/requirements/:projectKey', async (req, res) => {
  try {
    const requirements = await dataSourceManager.fetchJiraRequirements(req.params.projectKey);
    res.json(requirements);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.get('/api/confluence/pages/:spaceKey', async (req, res) => {
  try {
    const pages = await dataSourceManager.fetchConfluencePages(req.params.spaceKey);
    res.json(pages);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.get('/api/sharepoint/documents/:driveId', async (req, res) => {
  try {
    const documents = await dataSourceManager.fetchSharePointDocuments(req.params.driveId);
    res.json(documents);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// File upload
app.post('/api/upload', upload.array('files'), async (req, res) => {
  try {
    const processedFiles = [];
    
    for (const file of req.files) {
      const content = await DocumentProcessor.extractTextFromFile(file.path, file.mimetype);
      
      processedFiles.push({
        id: file.filename,
        name: file.originalname,
        content,
        source: 'upload',
        size: file.size
      });
    }
    
    // Add to vector store
    await userStoryGenerator.vectorStore.addDocuments(processedFiles);
    
    res.json({
      success: true,
      files: processedFiles.map(f => ({
        id: f.id,
        name: f.name,
        size: f.size
      }))
    });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Generate embeddings
app.post('/api/embeddings/generate', async (req, res) => {
  try {
    const { provider } = req.body;
    const client = clients[provider];
    
    if (!client) {
      return res.status(400).json({ error: `LLM provider ${provider} not configured` });
    }
    
    await userStoryGenerator.vectorStore.generateEmbeddings(client);
    
    res.json({
      success: true,
      documentCount: userStoryGenerator.vectorStore.documents.length
    });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Generate user stories
app.post('/api/generate-stories', async (req, res) => {
  try {
    const { requirements, llmConfig, dataSources } = req.body;
    
    // Collect all requirements from various sources
    let allRequirements = requirements || [];
    
    // Add requirements from connected data sources
    if (dataSources.jira && dataSources.jira.projectKey) {
      const jiraReqs = await dataSourceManager.fetchJiraRequirements(dataSources.jira.projectKey);
      allRequirements = allRequirements.concat(
        jiraReqs.map(req => `${req.summary}: ${req.description}`)
      );
      
      // Add to vector store
      await userStoryGenerator.vectorStore.addDocuments(
        jiraReqs.map(req => ({
          id: req.id,
          content: `${req.summary}\n${req.description}`,
          source: 'jira',
          title: req.summary
        }))
      );
    }
    
    if (dataSources.confluence && dataSources.confluence.spaceKey) {
      const confluencePages = await dataSourceManager.fetchConfluencePages(dataSources.confluence.spaceKey);
      
      // Add to vector store
      await userStoryGenerator.vectorStore.addDocuments(
        confluencePages.map(page => ({
          id: page.id,
          content: page.content,
          source: 'confluence',
          title: page.title
        }))
      );
    }
    
    if (dataSources.sharepoint && dataSources.sharepoint.driveId) {
      const sharepointDocs = await dataSourceManager.fetchSharePointDocuments(dataSources.sharepoint.driveId);
      
      // Add to vector store
      await userStoryGenerator.vectorStore.addDocuments(
        sharepointDocs.map(doc => ({
          id: doc.id,
          content: doc.content,
          source: 'sharepoint',
          title: doc.name
        }))
      );
    }
    
    // Generate embeddings if we have new documents
    if (userStoryGenerator.vectorStore.documents.length > 0) {
      const client = clients[llmConfig.provider];
      if (client) {
        await userStoryGenerator.vectorStore.generateEmbeddings(client);
      }
    }
    
    // Generate user stories
    const userStories = await userStoryGenerator.generateUserStories(allRequirements, llmConfig);
    
    res.json({
      success: true,
      userStories,
      metadata: {
        requirementsCount: allRequirements.length,
        documentsProcessed: userStoryGenerator.vectorStore.documents.length,
        llmProvider: llmConfig.provider,
        model: llmConfig.model
      }
    });
  } catch (error) {
    console.error('Error generating user stories:', error);
    res.status(500).json({ error: error.message });
  }
});

// Export to Jira
app.post('/api/export/jira', async (req, res) => {
  try {
    const { userStories, projectKey, issueTypeId } = req.body;
    
    const createdIssues = await dataSourceManager.exportToJira(
      userStories, 
      projectKey, 
      issueTypeId
    );
    
    res.json({
      success: true,
      createdIssues,
      count: createdIssues.length
    });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// LLM Configuration validation
app.post('/api/llm/validate', async (req, res) => {
  try {
    const { provider, apiKey, model } = req.body;
    
    // Temporarily set up client with provided API key
    let testClient;
    
    switch (provider) {
      case 'openai':
        testClient = new OpenAI({ apiKey });
        break;
      case 'azure':
        // Azure validation would require additional parameters
        return res.json({ valid: true, message: 'Azure configuration format accepted' });
      default:
        return res.status(400).json({ error: 'Unsupported provider' });
    }
    
    // Test with a simple completion
    const testResponse = await testClient.chat.completions.create({
      model: model || 'gpt-3.5-turbo',
      messages: [{ role: 'user', content: 'Test' }],
      max_tokens: 5
    });
    
    res.json({
      valid: true,
      message: 'API key validated successfully',
      model: testResponse.model
    });
  } catch (error) {
    res.status(400).json({
      valid: false,
      error: error.message
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: error.message
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`RAG User Stories API server running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});