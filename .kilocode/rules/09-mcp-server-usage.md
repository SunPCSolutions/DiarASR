# MCP Server Usage Guidelines

## Overview
This document teaches Roo Code how to effectively use Model Context Protocol (MCP) servers to extend its capabilities for complex software development tasks.

## MCP Server Architecture

### What are MCP Servers?
MCP servers are external tools that provide specialized capabilities through a standardized protocol:
- **Web Research**: Search, scrape, and analyze web content
- **Planning Tools**: Break down complex projects into manageable tasks
- **Reasoning Tools**: Structured thinking and problem analysis
- **Automation Tools**: Workflow orchestration and integration
- **Data Tools**: Vector search and knowledge base operations

### When to Use MCP Servers
Use MCP servers when:
- **Research is needed**: Technology choices, API documentation, best practices
- **Complex planning required**: Large projects needing systematic breakdown
- **External data required**: Web content, documentation, market research
- **Structured reasoning needed**: Complex problems requiring step-by-step analysis
- **Automation opportunities**: Repetitive tasks that can be orchestrated

## Available MCP Servers and Usage Patterns

### 1. Sequential Thinking MCP (`sequentialthinking`)

**Purpose**: Structured reasoning and problem analysis
**Best For**: Complex decision-making, requirement clarification, systematic problem solving

#### Usage Patterns
```typescript
// Use for requirement analysis
// 1. Break down complex user stories
// 2. Identify edge cases and constraints
// 3. Validate assumptions systematically

// Example workflow:
- Analyze user requirements step by step
- Identify technical constraints
- Evaluate solution alternatives
- Document decision rationale
```

#### When to Use
- **Requirement clarification**: Complex business logic that needs systematic breakdown
- **Architecture decisions**: Evaluating multiple technical approaches
- **Problem diagnosis**: Systematic troubleshooting of complex issues
- **Risk assessment**: Analyzing potential failure modes and mitigation strategies

### 2. Software Planning MCP (`software-planning-tool`)

**Purpose**: Project planning and task orchestration
**Best For**: Complex projects, milestone planning, dependency management

#### Usage Patterns
```typescript
// Use for project breakdown
// 1. Analyze project scope and complexity
// 2. Create structured task breakdown
// 3. Identify dependencies and blockers
// 4. Establish milestones and deliverables

// Integration with Orchestrator mode:
- Use planning MCP to create initial task breakdown
- Delegate subtasks to appropriate modes
- Track progress and adjust plans as needed
```

#### When to Use
- **Project initialization**: Breaking down large projects into manageable tasks
- **Milestone planning**: Establishing project phases and deliverables
- **Dependency mapping**: Understanding task relationships and constraints
- **Progress tracking**: Monitoring completion and adjusting timelines

### 3. Web Content Search MCP (`web-content-search-mcp`)

**Purpose**: Web research and content analysis
**Best For**: Technology research, documentation lookup, competitive analysis

#### Usage Patterns
```typescript
// Use for technology research
// 1. Search for technology comparisons and reviews
// 2. Find API documentation and examples
// 3. Research security best practices
// 4. Analyze industry trends and standards

// Example research workflow:
- Search for "REST API authentication best practices 2024"
- Scrape official documentation sites
- Compare multiple technology options
- Summarize findings for team decisions
```

#### When to Use
- **Technology evaluation**: Comparing frameworks, libraries, or platforms
- **API integration**: Finding documentation and usage examples
- **Security research**: Latest vulnerabilities and mitigation strategies
- **Industry standards**: Current best practices and compliance requirements

### 4. Qdrant MCP (`qdrant-mcp`)

**Purpose**: Vector database operations and semantic search
**Best For**: Knowledge base queries, document similarity, AI-powered search

#### Usage Patterns
```typescript
// Use for knowledge retrieval
// 1. Search similar code patterns or solutions
// 2. Find relevant documentation or examples
// 3. Query project knowledge base
// 4. Semantic search across large codebases

// Example usage:
- Search for "error handling patterns in Node.js APIs"
- Find similar implementations in existing codebase
- Query documentation for specific technologies
```

#### When to Use
- **Code reuse**: Finding similar patterns in existing codebases
- **Knowledge retrieval**: Accessing project documentation and best practices
- **Pattern recognition**: Identifying common solutions to recurring problems
- **Documentation search**: Finding relevant technical documentation

### 5. Context7 MCP (`context7`)

**Purpose**: Context-aware code analysis and understanding
**Best For**: Code comprehension, refactoring assistance, pattern recognition

#### Usage Patterns
```typescript
// Use for code analysis
// 1. Understand complex codebases quickly
// 2. Identify refactoring opportunities
// 3. Analyze code patterns and anti-patterns
// 4. Generate insights about code quality

// Integration with development:
- Analyze new codebases before making changes
- Identify areas needing refactoring
- Understand legacy code patterns
- Generate code quality insights
```

#### When to Use
- **Code review**: Analyzing code quality and patterns
- **Refactoring planning**: Identifying improvement opportunities
- **Onboarding**: Understanding new codebases quickly
- **Code quality assessment**: Generating insights about maintainability

### 6. Puppeteer MCP (`puppeteer`)

**Purpose**: Browser automation and web interaction
**Best For**: Web application testing, content extraction, UI automation

#### Usage Patterns
```typescript
// Use for web automation
// 1. Test web application user flows
// 2. Extract data from web pages
// 3. Automate repetitive web tasks
// 4. Validate web application behavior

// Example testing workflow:
- Navigate to application login page
- Fill out and submit login form
- Verify successful authentication
- Test protected routes and functionality
```

#### When to Use
- **E2E testing**: Automating user journey tests
- **Content extraction**: Scraping data from web pages
- **UI validation**: Testing web application interfaces
- **Integration testing**: Validating external web service integrations

### 7. N8N MCP (`n8n-mcp`)

**Purpose**: Workflow automation and API orchestration
**Best For**: Complex integrations, data processing workflows, automation

#### Usage Patterns
```typescript
// Use for workflow automation
// 1. Design complex data processing pipelines
// 2. Orchestrate multiple API calls
// 3. Automate business processes
// 4. Create integration workflows

// Example automation:
- Process user registration data
- Validate against multiple services
- Send confirmation emails
- Update multiple databases
```

#### When to Use
- **API orchestration**: Coordinating multiple service calls
- **Data processing**: Complex ETL operations and transformations
- **Business automation**: Streamlining repetitive business processes
- **Integration workflows**: Connecting disparate systems and services

### 8. Crawlee MCP (`crawlee-mcp`)

**Purpose**: Advanced web scraping and crawling
**Best For**: Large-scale data collection, website monitoring, content aggregation

#### Usage Patterns
```typescript
// Use for large-scale scraping
// 1. Crawl entire websites for data collection
// 2. Monitor website changes over time
// 3. Aggregate content from multiple sources
// 4. Handle complex scraping scenarios

// Example usage:
- Scrape product data from e-commerce sites
- Monitor competitor pricing changes
- Collect research data from academic sites
- Aggregate news and content feeds
```

#### When to Use
- **Data collection**: Large-scale web scraping projects
- **Market research**: Competitive analysis and pricing data
- **Content aggregation**: Collecting data from multiple web sources
- **Monitoring**: Tracking website changes and updates

## Integration Patterns with Roo Code Modes

### Architect Mode + MCP Servers
```markdown
# System Design with Research
1. Use web-content-search-mcp to research technology options
2. Use sequentialthinking to evaluate architectural decisions
3. Use software-planning-tool to break down implementation phases
4. Document findings in system design documents
```

### Code Mode + MCP Servers
```typescript
// Implementation with Research
// 1. Use qdrant-mcp to find similar code patterns
// 2. Use web-content-search-mcp for API documentation
// 3. Use context7 for code quality insights
// 4. Implement following established patterns
```

### Orchestrator Mode + MCP Servers
```markdown
# Complex Project Management
1. Use software-planning-tool for initial project breakdown
2. Use sequentialthinking for complex task analysis
3. Delegate research tasks using web-content-search-mcp
4. Coordinate automation using n8n-mcp
```

### Debug Mode + MCP Servers
```typescript
// Systematic Troubleshooting
// 1. Use sequentialthinking for problem analysis
// 2. Use web-content-search-mcp for similar issue research
// 3. Use qdrant-mcp to find related code patterns
// 4. Document solutions in debughistory.md
```

## Best Practices for MCP Server Usage

### Efficiency Guidelines
- **Use appropriate tools**: Choose the right MCP server for each task
- **Combine capabilities**: Use multiple MCP servers for complex workflows
- **Cache results**: Store research findings in Memory Bank
- **Document usage**: Record successful MCP server applications

### Quality Assurance
- **Validate results**: Always verify MCP server outputs
- **Cross-reference**: Compare findings from multiple sources
- **Update knowledge**: Keep Memory Bank current with new findings
- **Share insights**: Document successful patterns for team use

### Performance Considerations
- **Timeout awareness**: Respect MCP server timeout settings
- **Resource management**: Be mindful of rate limits and quotas
- **Fallback strategies**: Have alternatives when MCP servers are unavailable
- **Cost optimization**: Use MCP servers efficiently to minimize resource usage

### Security and Compliance
- **Data privacy**: Ensure MCP server usage complies with data protection rules
- **Access control**: Use appropriate authentication for sensitive operations
- **Audit trail**: Document MCP server usage for compliance purposes
- **Secure storage**: Protect any sensitive data retrieved via MCP servers

## Common MCP Server Workflows

### Technology Research Workflow
1. **Define research question** clearly
2. **Use web-content-search-mcp** to gather information
3. **Use sequentialthinking** to analyze findings
4. **Document conclusions** in techContext.md

### Project Planning Workflow
1. **Assess project complexity** and scope
2. **Use software-planning-tool** for task breakdown
3. **Use sequentialthinking** for risk analysis
4. **Create implementation plan** with milestones

### Problem Solving Workflow
1. **Define problem** and constraints clearly
2. **Use sequentialthinking** for systematic analysis
3. **Use web-content-search-mcp** for similar solutions
4. **Use qdrant-mcp** for code pattern matching

### Integration Testing Workflow
1. **Design test scenarios** using sequentialthinking
2. **Use puppeteer** for UI automation testing
3. **Use n8n-mcp** for API orchestration testing
4. **Document test results** and issues found

By following these guidelines, Roo Code will effectively leverage MCP servers to enhance its capabilities for complex software development tasks, research, planning, and problem-solving.