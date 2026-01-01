# Project Structure Guidelines

## Overview
This document teaches Roo Code how to organize complex software projects following industry best practices for maintainability, scalability, and team collaboration.

## Core Principles

### 1. Separation of Concerns
- **Single Responsibility**: Each module/component should have one clear purpose
- **Layered Architecture**: Separate business logic, data access, and presentation layers
- **Modular Design**: Break down complex systems into manageable, independent modules

### 2. Scalable Organization
- **Feature-Based Structure**: Organize by business features rather than technical layers
- **Domain-Driven Design**: Structure around business domains and bounded contexts
- **Evolutionary Architecture**: Design for easy extension and modification

### 3. Developer Experience
- **Intuitive Navigation**: Clear, logical folder structures
- **Consistent Patterns**: Predictable organization across projects
- **Tool-Friendly**: Structure that works well with development tools

## Directory Structure Patterns

### Web Application Structure
```
project-root/
├── src/                          # Source code
│   ├── components/               # Reusable UI components
│   ├── pages/                    # Page-level components
│   ├── hooks/                    # Custom React hooks
│   ├── utils/                    # Utility functions
│   ├── services/                 # API and external service integrations
│   ├── types/                    # TypeScript type definitions
│   ├── constants/                # Application constants
│   └── styles/                   # Global styles and themes
├── public/                       # Static assets
├── tests/                        # Test files
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
├── docs/                         # Documentation
├── scripts/                      # Build and utility scripts
├── config/                       # Configuration files
└── .roo/                         # Roo Code instructions
```

### API Backend Structure
```
project-root/
├── src/
│   ├── controllers/              # Request handlers
│   ├── models/                   # Data models and schemas
│   ├── routes/                   # API route definitions
│   ├── middleware/               # Custom middleware
│   ├── services/                 # Business logic services
│   ├── utils/                    # Utility functions
│   ├── config/                   # Application configuration
│   └── types/                    # Type definitions
├── tests/
├── docs/
├── scripts/
├── migrations/                   # Database migrations
└── .roo/
```

### Microservices Structure
```
project-root/
├── services/                     # Individual microservices
│   ├── user-service/
│   ├── order-service/
│   └── notification-service/
├── infrastructure/               # Infrastructure as code
├── docker/                       # Docker configurations
├── k8s/                          # Kubernetes manifests
├── monitoring/                   # Monitoring and logging
└── .roo/
```

## File Organization Rules

### Naming Conventions
- **kebab-case** for directories: `user-management`, `order-processing`
- **PascalCase** for components: `UserProfile.tsx`, `OrderForm.tsx`
- **camelCase** for utilities: `formatDate.ts`, `validateEmail.ts`
- **UPPER_SNAKE_CASE** for constants: `API_BASE_URL`, `MAX_RETRY_COUNT`

### File Grouping
- **Related files together**: Keep components, styles, and tests near each other
- **Index files**: Use `index.ts` files for clean imports
- **Barrel exports**: Export related functionality from index files

### Import Organization
```typescript
// 1. External dependencies
import React from 'react';
import { useState } from 'react';

// 2. Internal absolute imports
import { User } from '@/types/user';
import { formatDate } from '@/utils/date';

// 3. Internal relative imports
import { Button } from '../components/Button';
import { api } from '../../services/api';
```

## Architecture Patterns

### Feature-Based Organization
```
src/
├── features/
│   ├── authentication/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── types/
│   │   └── tests/
│   ├── dashboard/
│   └── user-profile/
```

### Domain-Driven Design
```
src/
├── domains/
│   ├── user/
│   │   ├── entities/
│   │   ├── value-objects/
│   │   ├── repositories/
│   │   └── services/
│   ├── order/
│   └── product/
```

### Clean Architecture
```
src/
├── domain/                       # Business entities and rules
├── application/                  # Use cases and application logic
├── infrastructure/               # External concerns (DB, APIs, etc.)
└── presentation/                 # UI and controllers
```

## Configuration Management

### Environment Variables
- **Naming**: `REACT_APP_`, `API_`, `DB_` prefixes
- **Documentation**: Document all required environment variables
- **Defaults**: Provide sensible defaults for development

### Configuration Files
- **Centralized**: Keep configuration in dedicated config directory
- **Environment-specific**: Separate configs for dev/staging/prod
- **Validation**: Validate configuration on startup

## Tool Integration

### Build Tools
- **Consistent scripts**: Standard npm scripts across projects
- **Configuration files**: Place in project root or config directory
- **Caching**: Configure build caches for faster development

### Development Tools
- **Editor config**: `.editorconfig` for consistent formatting
- **Pre-commit hooks**: Enforce code quality before commits
- **Linting**: Configure ESLint, Prettier, TypeScript

## Scaling Considerations

### Monorepo Structure
```
monorepo/
├── packages/
│   ├── ui-components/
│   ├── api-client/
│   └── utils/
├── apps/
│   ├── web-app/
│   ├── mobile-app/
│   └── admin-panel/
└── tools/
```

### Performance Optimization
- **Code splitting**: Organize for efficient bundling
- **Lazy loading**: Structure for dynamic imports
- **Asset optimization**: Organize static assets for caching

## Documentation Structure

### Code Documentation
- **README files**: In each major directory
- **API docs**: Document public interfaces
- **Architecture docs**: Explain system design decisions

### Project Documentation
```
docs/
├── architecture/                 # System design docs
├── api/                          # API documentation
├── development/                  # Development guides
└── deployment/                   # Deployment instructions
```

## Quality Assurance

### Testing Structure
- **Test files**: Co-located with implementation files
- **Test utilities**: Shared testing helpers and mocks
- **Coverage reports**: Configure coverage thresholds

### Code Quality
- **Linting rules**: Consistent code style enforcement
- **Type checking**: Strict TypeScript configuration
- **Pre-commit hooks**: Automated quality checks

## Migration and Refactoring

### Safe Refactoring
- **Incremental changes**: Small, testable modifications
- **Feature flags**: Use feature toggles for gradual rollouts
- **Backwards compatibility**: Maintain API contracts during changes

### Legacy Code Handling
- **Strangler pattern**: Gradually replace legacy components
- **Adapter pattern**: Wrap legacy code with modern interfaces
- **Documentation**: Document technical debt and migration plans

## Best Practices Summary

1. **Plan before coding**: Design structure based on project requirements
2. **Consistent conventions**: Follow established naming and organization patterns
3. **Modular design**: Create reusable, independent components
4. **Scalable architecture**: Design for growth and change
5. **Quality focus**: Integrate testing and code quality from the start
6. **Documentation**: Keep code and architecture well-documented
7. **Tool integration**: Leverage development tools for productivity

## Common Patterns to Avoid

- **God objects**: Large classes with multiple responsibilities
- **Tight coupling**: Highly interdependent modules
- **Deep nesting**: Overly nested directory structures
- **Inconsistent naming**: Mixed naming conventions
- **Large files**: Files that are difficult to understand and maintain
- **Undocumented code**: Code without clear purpose or usage

By following these guidelines, Roo Code will create well-organized, maintainable, and scalable software projects that follow industry best practices.