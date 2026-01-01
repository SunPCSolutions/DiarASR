# Development Workflow Algorithm

## Overview
This document provides a structured algorithm for Roo Code to follow throughout the software development lifecycle. It specifies which files to review at different stages and under what conditions.

## 🔄 Workflow Algorithm

### Phase 1: Project Initialization
**Trigger**: New project or first session
**Review Files**:
- `memory-bank/projectbrief.md` - Project goals and scope
- `memory-bank/productContext.md` - User problems and requirements
- `memory-bank/techContext.md` - Technology stack and constraints
- `.roo/rules/project-structure.md` - Directory organization
- `.roo/rules/coding-standards.md` - Development standards

### Phase 2: Architecture & Planning
**Trigger**: Starting system design or major feature planning
**Review Files**:
- `.roo/rules-architect/system-design.md` - Architecture guidelines
- `memory-bank/systemPatterns.md` - Existing patterns
- `memory-bank/activeContext.md` - Current work status

**Conditional Reviews**:
- **Web Application**: `web-accessibility.md`, `responsive-design.md`
- **API Development**: `mcp-server-usage.md` for API orchestration
- **Data-Intensive**: Focus on database patterns in system-design.md

### Phase 3: Implementation
**Trigger**: Starting code development
**Review Files**:
- `.roo/rules-code/implementation.md` - Implementation guidelines
- `.roo/rules/testing-strategy.md` - Testing approach
- `.roo/rules/documentation.md` - Documentation standards

**Conditional Reviews**:
- **Frontend Work**: `inclusive-design.md` for UX considerations
- **Backend Work**: `mcp-server-usage.md` for integrations
- **Complex Logic**: `systemPatterns.md` for established patterns

### Phase 4: Quality Assurance
**Trigger**: Pre-deployment or feature completion
**Review Files**:
- `.roo/rules/testing-strategy.md` - Comprehensive testing
- `memory-bank/progress.md` - Completion verification
- `VALIDATION.md` - Quality checklist

**Conditional Reviews**:
- **Web Deployment**: `web-accessibility.md` for compliance
- **API Deployment**: `mcp-server-usage.md` for monitoring
- **User-Facing**: `inclusive-design.md` for user experience

### Phase 5: Maintenance & Debugging
**Trigger**: Issues, bugs, or system changes
**Review Files**:
- `memory-bank/debughistory.md` - Past issue patterns
- `memory-bank/activeContext.md` - Current system state
- `.roo/rules/testing-strategy.md` - Debugging approaches

## 🎯 Context-Aware Reviews

### By Technology Stack
```yaml
# Web Applications
- web-accessibility.md
- responsive-design.md
- inclusive-design.md

# APIs/Microservices
- mcp-server-usage.md
- system-design.md (communication patterns)
- testing-strategy.md (integration testing)

# Data Systems
- system-design.md (data patterns)
- implementation.md (performance)
- documentation.md (schema docs)

# Mobile Applications
- responsive-design.md (touch targets)
- inclusive-design.md (accessibility)
- testing-strategy.md (device testing)
```

### By Development Phase
```yaml
# Planning Phase
- projectbrief.md, productContext.md
- system-design.md
- techContext.md

# Development Phase
- implementation.md
- coding-standards.md
- testing-strategy.md

# Testing Phase
- testing-strategy.md
- documentation.md
- progress.md

# Deployment Phase
- system-design.md (deployment patterns)
- mcp-server-usage.md (monitoring)
- VALIDATION.md
```

### By Issue Type
```yaml
# Performance Issues
- implementation.md (performance section)
- system-design.md (scalability)
- testing-strategy.md (load testing)

# Security Issues
- coding-standards.md (security)
- system-design.md (security architecture)
- mcp-server-usage.md (secure integrations)

# User Experience Issues
- inclusive-design.md
- web-accessibility.md
- responsive-design.md
```

## 🔀 Mode Selection Algorithm

### Automatic Mode Switching
```typescript
// Based on task analysis
function selectMode(task: string): Mode {
  if (task.includes('design') || task.includes('architecture')) {
    return 'architect';
  }
  if (task.includes('debug') || task.includes('fix') || task.includes('error')) {
    return 'debug';
  }
  if (task.includes('explain') || task.includes('understand')) {
    return 'ask';
  }
  if (task.includes('complex') || task.includes('multiple') || task.includes('orchestrate')) {
    return 'orchestrator';
  }
  return 'code'; // Default for implementation
}
```

### Custom Mode Integration
- **Rule Directories**: Custom modes automatically get `.roo/rules-{modeSlug}/` directories
- **AGENTS.md Support**: Use `AGENTS.md` or `AGENT.md` in workspace root for agent-specific rules
- **Mode-Specific Instructions**: Combine with prompts tab and rule files for comprehensive customization

### AGENTS.md Support
- **Location**: `AGENTS.md` or `AGENT.md` in workspace root
- **Purpose**: Agent-specific rules and guidelines for AI behavior
- **Loading**: Automatic (can be disabled with `"roo-cline.useAgentRules": false`)
- **Priority**: Loaded after mode-specific rules, before general workspace rules
- **Use Case**: Team standards for AI agent behavior, version-controlled alongside project code

### Mode-Specific File Reviews
```yaml
# Architect Mode
- system-design.md
- project-structure.md
- techContext.md

# Code Mode
- implementation.md
- coding-standards.md
- testing-strategy.md

# Debug Mode
- debughistory.md
- testing-strategy.md (debugging section)
- activeContext.md

# Ask Mode
- documentation.md
- systemPatterns.md
- Any relevant rule files for explanation
```

## 📊 Progress Tracking Integration

### Memory Bank Updates
- **Start of Task**: Update `activeContext.md` with current work
- **Milestone Completion**: Update `progress.md` with achievements
- **Issue Discovery**: Document in `debughistory.md`
- **Pattern Identification**: Add to `systemPatterns.md`

### Validation Checkpoints
```yaml
# Pre-Implementation
- [ ] Requirements reviewed (productContext.md)
- [ ] Architecture approved (systemPatterns.md)
- [ ] Standards confirmed (coding-standards.md)

# Mid-Implementation
- [ ] Tests written (testing-strategy.md)
- [ ] Documentation updated (documentation.md)
- [ ] Progress tracked (progress.md)

# Post-Implementation
- [ ] Code reviewed (coding-standards.md)
- [ ] Tests passing (testing-strategy.md)
- [ ] Documentation complete (documentation.md)
```

## 🔧 Tool Usage Guidelines

### By Development Phase
```yaml
# Planning Phase (Architect Mode)
- read_file: Analyze requirements and existing code
- grep_search: Find patterns and dependencies
- run_terminal_cmd: Architecture analysis tools

# Implementation Phase (Code Mode)
- apply_diff: Surgical code changes
- search_replace: Targeted updates
- run_terminal_cmd: Build, test, and lint commands

# Testing Phase (Code/Debug Mode)
- run_terminal_cmd: Test execution and coverage
- read_file: Analyze test results
- apply_diff: Fix identified issues

# Documentation Phase (Code Mode)
- read_file: Review existing documentation
- apply_diff: Update documentation files
- run_terminal_cmd: Documentation generation
```

## 🎯 Decision Tree Algorithm

```
Start Task
├── Is this a new project?
│   ├── Yes → Phase 1: Project Initialization
│   └── No → Analyze task type
│       ├── Architecture/Design → Phase 2: Architecture & Planning
│       ├── Code Implementation → Phase 3: Implementation
│       ├── Quality/Testing → Phase 4: Quality Assurance
│       └── Bug Fixes/Maintenance → Phase 5: Maintenance & Debugging
│
├── Select appropriate mode based on task
├── Review phase-specific files
├── Apply conditional reviews based on technology/project type
├── Execute task using reviewed guidelines
├── Update Memory Bank with results
└── Validate against quality checkpoints
```

## 📈 Continuous Learning Integration

### Pattern Recognition
- **Similar Tasks**: Reference `systemPatterns.md` for established approaches
- **Past Issues**: Check `debughistory.md` for similar problem patterns
- **Successful Solutions**: Review `progress.md` for proven methodologies

### Knowledge Accumulation
- **New Patterns**: Document in `systemPatterns.md`
- **Lessons Learned**: Add to `debughistory.md`
- **Best Practices**: Update relevant rule files

This algorithm ensures Roo Code follows a systematic, context-aware approach to development, reviewing the right files at the right time for optimal efficiency and quality.