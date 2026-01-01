# Code Mode: Implementation Guidelines

## Tool Usage Strategy

### Read Tools - Understand First
```typescript
// Explore codebase structure
list_dir("src")
read_file("src/services/userService.ts")
grep_search("class UserService", "src/**/*.ts")
```

### Edit Tools - Surgical Changes
```typescript
// ✅ Targeted edits only
apply_diff("src/services/userService.ts", {
  search: "  async updateUser(id: string, data: UpdateUserData): Promise<User> {\n    // Implementation\n  }\n}",
  replace: "  async updateUser(id: string, data: UpdateUserData): Promise<User> {\n    // Implementation\n  }\n\n  async deleteUser(id: string): Promise<void> {\n    const user = await this.findById(id);\n    if (!user) throw new NotFoundError('User not found');\n    await this.userRepository.delete(id);\n  }\n}"
});

// ❌ Avoid replacing entire files
```

### Command Tools - Verification Only
```bash
# ✅ Run tests and checks
npm test -- --testNamePattern="UserService"
npm run lint src/services/userService.ts
npm run type-check

# ❌ Avoid destructive operations
```

## Implementation Workflow

### 1. Understand Requirements
```typescript
read_file("memory-bank/productContext.md")
read_file("memory-bank/activeContext.md")
grep_search("TODO|FIXME", "src/**/*")
```

### 2. Plan Implementation
- Identify affected components
- Plan data flow and error handling
- Design interfaces and contracts

### 3. Implement Incrementally
```typescript
// Start with types
apply_diff("src/types/user.ts", { /* Add types */ });

// Core logic
apply_diff("src/services/userService.ts", { /* Service methods */ });

// Tests
apply_diff("src/services/userService.test.ts", { /* Test cases */ });

// Dependent components
apply_diff("src/controllers/userController.ts", { /* Controller updates */ });
```

### 4. Verify Implementation
```bash
npm test src/services/userService.test.ts
npm run lint src/services/
npm run type-check
npm run test:integration
```

## Code Quality Standards

### Error Handling
```typescript
// ✅ Specific error types
export class ValidationError extends Error {
  constructor(field: string, message: string) {
    super(`Validation failed for ${field}: ${message}`);
    this.name = 'ValidationError';
  }
}

async function createUser(data: CreateUserData): Promise<User> {
  if (!data.email?.includes('@')) {
    throw new ValidationError('email', 'Invalid email format');
  }

  try {
    return await userRepository.create(data);
  } catch (error) {
    if (error.code === 'DUPLICATE_KEY') {
      throw new ValidationError('email', 'Email already exists');
    }
    throw error;
  }
}
```

### Type Safety
```typescript
// ✅ Strict typing with generics
interface Repository<T> {
  findById(id: string): Promise<T | null>;
  findAll(filter?: Partial<T>): Promise<T[]>;
  create(data: Omit<T, 'id' | 'createdAt'>): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}

class UserRepository implements Repository<User> {
  // Full type safety
}
```

## Testing Implementation

### Test-Driven Development
```typescript
// 1. Write failing test
it('should create user with valid data', async () => {
  const userData = { name: 'John', email: 'john@example.com' };
  mockRepository.create.mockResolvedValue(expectedUser);

  const result = await userService.createUser(userData);

  expect(result).toEqual(expectedUser);
});

// 2. Implement minimal code
async createUser(data: CreateUserData): Promise<User> {
  return await this.userRepository.create(data);
}

// 3. Refactor with validation
async createUser(data: CreateUserData): Promise<User> {
  this.validateUserData(data);
  return await this.userRepository.create({
    ...data,
    createdAt: new Date(),
  });
}
```

### Coverage Requirements
- **Unit Tests**: All public methods and error paths
- **Integration Tests**: Component interactions
- **Edge Cases**: Null/undefined, empty arrays, special characters

## Refactoring Techniques

### Extract Methods
```typescript
// ✅ Break down large methods
class OrderService {
  async processOrder(orderData: OrderData): Promise<Order> {
    const validatedData = this.validateOrderData(orderData);
    const user = await this.getUserWithPermissions(validatedData.userId);
    const items = await this.reserveInventory(validatedData.items);
    const total = this.calculateTotal(items, validatedData.discounts);

    return await this.createOrder({
      ...validatedData,
      userId: user.id,
      items,
      total,
    });
  }

  private validateOrderData(data: OrderData): ValidatedOrderData { /* ... */ }
  private async getUserWithPermissions(userId: string): Promise<User> { /* ... */ }
}
```

### Improve Naming
```typescript
// ✅ Descriptive names
class PaymentProcessor {
  async processCreditCardPayment(
    cardDetails: CardDetails,
    amount: Money,
    orderId: string
  ): Promise<PaymentResult> { /* ... */ }
}

// ❌ Poor names
class PayProc {
  async procCC(card, amt, ordId) { /* ... */ }
}
```

## Performance Considerations

### Efficient Algorithms
```typescript
// ✅ O(n log n) sorting
function findTopItems(items: Item[], limit: number): Item[] {
  return items
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

// ❌ O(n²) nested loops
function findTopItems(items, limit) {
  const result = [];
  for (let i = 0; i < limit; i++) {
    // Inefficient nested loop
  }
  return result;
}
```

### Memory Management
```typescript
// ✅ Streaming for large files
async function* processLargeFile(filePath: string): AsyncIterable<ProcessedData> {
  const stream = fs.createReadStream(filePath, { encoding: 'utf8' });
  for await (const chunk of stream) {
    const processed = await processChunk(chunk);
    yield processed;
  }
}

// ❌ Loading entire file
async function processLargeFile(filePath: string): Promise<ProcessedData[]> {
  const content = await fs.promises.readFile(filePath, 'utf8');
  const lines = content.split('\n');
  return lines.map(processLine); // Memory intensive
}
```

## Security Best Practices

### Input Validation
```typescript
function sanitizeUserInput(input: string): string {
  return input
    .replace(/[<>]/g, '')  // Remove HTML tags
    .trim()
    .substring(0, 1000);   // Limit length
}

function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email) && email.length <= 254;
}
```

### Secure Coding
- Use parameterized queries (prevent SQL injection)
- Validate file uploads (type and size checks)
- Sanitize user input
- Use HTTPS for data transmission
- Implement proper authentication

## Code Review Preparation

### Self-Review Checklist
- [ ] Code follows established patterns
- [ ] Functions have single responsibility
- [ ] Error cases handled appropriately
- [ ] Tests comprehensive and passing
- [ ] Documentation clear and complete
- [ ] Security considerations addressed
- [ ] Performance acceptable
- [ ] Code readable and maintainable

### Common Issues
- Magic numbers → named constants
- Long methods → extract functions
- Deep nesting → simplify logic
- Code duplication → extract common code
- Missing validation → add input checks
- Inconsistent naming → use conventions

Focus on incremental implementation, comprehensive testing, and clean, maintainable code that follows established patterns and security best practices.