# Coding Standards and Best Practices

## Core Principles

### Clean Code
- **Readable**: Easy to read and understand
- **Maintainable**: Easy to modify and extend
- **Testable**: Well-structured for testing
- **Efficient**: Performant and resource-conscious

### SOLID Principles
- **Single Responsibility**: One reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Subtypes substitutable for base types
- **Interface Segregation**: No unused interface dependencies
- **Dependency Inversion**: Depend on abstractions

### DRY (Don't Repeat Yourself)
- Eliminate duplication
- Extract common functionality
- Create reusable abstractions

## Language Standards

### TypeScript/JavaScript

#### Naming Conventions
```typescript
// ✅ Good: Descriptive, camelCase
const userProfile = { name: 'John', age: 30 };
function calculateTotalPrice(items: CartItem[]): number { ... }

// ❌ Bad: Non-descriptive, inconsistent
const x = { n: 'John', a: 30 };
function calc(items) { ... }
```

#### Type Safety
```typescript
// ✅ Explicit types, strict null checks
interface User {
  id: string;
  name: string;
  email?: string;
}

function createUser(data: Partial<User>): User {
  if (!data.name) throw new Error('Name is required');
  return { id: generateId(), ...data };
}

// ❌ Any types, implicit any
function createUser(data) {
  return { id: Math.random(), ...data };
}
```

#### Error Handling
```typescript
// ✅ Specific error types
class ValidationError extends Error {
  constructor(field: string, message: string) {
    super(`Validation failed for ${field}: ${message}`);
    this.name = 'ValidationError';
  }
}

function validateEmail(email: string): void {
  if (!email.includes('@')) {
    throw new ValidationError('email', 'Invalid email format');
  }
}

// ❌ Generic errors, silent failures
function validateEmail(email) {
  if (!email.includes('@')) {
    console.log('Invalid email');
    return false;
  }
}
```

### Python

#### PEP 8 Compliance
```python
# ✅ Proper naming, spacing, line length
def calculate_total_price(items: List[Dict[str, Any]]) -> float:
    """Calculate total price of items in cart."""
    total = 0.0
    for item in items:
        if 'price' in item and 'quantity' in item:
            total += item['price'] * item['quantity']
    return total

# ❌ Poor naming, no type hints
def calc(items):
    total=0
    for item in items:
        total+=item['price']*item['quantity'] if 'price' in item else 0
    return total
```

## Code Organization

### Function Design
```typescript
// ✅ Single responsibility, clear interface
function sendWelcomeEmail(user: User): Promise<void> {
  const subject = 'Welcome to our platform!';
  const body = generateWelcomeTemplate(user);
  return emailService.send(user.email, subject, body);
}

// ❌ Multiple responsibilities
function handleUser(user) {
  // Create user, send email, log analytics, update database
}
```

### Class Design
```typescript
// ✅ Single responsibility, dependency injection
class UserService {
  constructor(
    private readonly userRepository: UserRepository,
    private readonly emailService: EmailService
  ) {}

  async createUser(userData: CreateUserData): Promise<User> {
    const user = await this.userRepository.create(userData);
    await this.emailService.sendWelcomeEmail(user);
    return user;
  }
}

// ❌ God class, tight coupling
class UserManager {
  createUser(userData) {
    // Mixed database, email, validation, analytics logic
  }
}
```

## Testing Standards

### Unit Testing
```typescript
// Arrange-Act-Assert pattern
describe('UserService', () => {
  let userService: UserService;
  let mockRepository: jest.Mocked<UserRepository>;

  beforeEach(() => {
    mockRepository = {
      create: jest.fn(),
      findById: jest.fn(),
    };
    userService = new UserService(mockRepository);
  });

  it('should create user successfully', async () => {
    const userData = { name: 'John', email: 'john@example.com' };
    const expectedUser = { id: '1', ...userData };

    mockRepository.create.mockResolvedValue(expectedUser);

    const result = await userService.createUser(userData);

    expect(result).toEqual(expectedUser);
    expect(mockRepository.create).toHaveBeenCalledWith(userData);
  });
});
```

## Documentation Standards

### Code Comments
```typescript
// ✅ Explains why, not what
/**
 * Calculates total price with tax. Uses progressive tax rates:
 * - 0-1000: 5% tax
 * - 1000+: 10% tax
 */
function calculateTotalWithTax(subtotal: number): number {
  const taxRate = subtotal > 1000 ? 0.10 : 0.05;
  return subtotal * (1 + taxRate);
}

// ❌ Obvious comments
// This function calculates the total
function calculateTotal(subtotal: number): number {
  // Add tax to subtotal
  return subtotal * 1.05;
}
```

### API Documentation
```typescript
/**
 * User authentication service
 * Handles login, logout, and session management
 */
export class AuthService {
  /**
   * Authenticates user with email and password
   * @param credentials - User login credentials
   * @returns Authentication result
   * @throws {AuthenticationError} When credentials invalid
   */
  async login(credentials: LoginCredentials): Promise<AuthResult> {
    // Implementation
  }
}
```

## Performance Considerations

### Efficient Algorithms
```typescript
// ✅ O(n) complexity
function findDuplicates(items: number[]): number[] {
  const seen = new Set<number>();
  const duplicates = new Set<number>();

  for (const item of items) {
    if (seen.has(item)) {
      duplicates.add(item);
    } else {
      seen.add(item);
    }
  }

  return Array.from(duplicates);
}

// ❌ O(n²) complexity
function findDuplicates(items: number[]): number[] {
  const duplicates = [];
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      if (items[i] === items[j] && !duplicates.includes(items[i])) {
        duplicates.push(items[i]);
      }
    }
  }
  return duplicates;
}
```

### Memory Management
- Avoid memory leaks (clean up event listeners, timers)
- Use efficient data structures
- Implement lazy loading
- Cache expensive operations

## Security Practices

### Input Validation
```typescript
function createPost(data: CreatePostData): Post {
  if (!data.title?.trim()) {
    throw new ValidationError('title', 'Title is required');
  }
  if (data.title.length > 200) {
    throw new ValidationError('title', 'Title too long');
  }

  // Sanitize HTML content
  const sanitizedContent = sanitizeHtml(data.content);

  return {
    id: generateId(),
    title: data.title.trim(),
    content: sanitizedContent,
    authorId: data.authorId,
    createdAt: new Date(),
  };
}
```

### Secure Coding
- Use parameterized queries (prevent SQL injection)
- Sanitize user input (prevent XSS)
- Implement CSRF protection
- Use proper authentication and session management

## Code Review Checklist

### Functionality
- [ ] Code meets requirements
- [ ] Edge cases handled
- [ ] Error conditions managed
- [ ] Security considerations addressed

### Code Quality
- [ ] Follows coding standards
- [ ] Readable and maintainable
- [ ] Proper documentation
- [ ] Adequate test coverage

### Performance
- [ ] Efficient algorithms used
- [ ] No memory leaks
- [ ] Reasonable resource usage
- [ ] Scalable design

### Best Practices
- [ ] SOLID principles followed
- [ ] DRY principle maintained
- [ ] Appropriate design patterns used
- [ ] Framework conventions followed

Follow these standards to produce high-quality, maintainable, and scalable software that meets industry best practices.