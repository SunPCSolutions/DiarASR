# Web Accessibility Guidelines

## WCAG 2.1 Principles

### 1. Perceivable
- **Alt text**: All images need descriptive `alt` attributes
- **Color contrast**: 4.5:1 ratio for normal text, 3:1 for large text
- **Color independence**: Don't rely on color alone to convey information
- **Media alternatives**: Captions for video, transcripts for audio

### 2. Operable
- **Keyboard accessible**: All interactive elements reachable via keyboard
- **Focus visible**: Clear focus indicators with good contrast
- **Timing controllable**: Users can pause/stop time-limited content
- **No keyboard traps**: Users can navigate away from any element

### 3. Understandable
- **Clear language**: Simple, jargon-free content
- **Predictable navigation**: Consistent behavior across pages
- **Input assistance**: Clear labels, error messages, and help text
- **Error prevention**: Confirmation for destructive actions

### 4. Robust
- **Semantic HTML**: Use proper elements (`<main>`, `<nav>`, `<article>`)
- **Heading hierarchy**: Logical h1→h2→h3 structure
- **ARIA sparingly**: Only when semantic HTML insufficient
- **Standards compliant**: Valid HTML, CSS, and JavaScript

## Essential Implementation

### Semantic HTML Structure
```html
<!-- ✅ Good -->
<main>
  <header>
    <h1>Page Title</h1>
    <nav aria-label="Main navigation">
      <ul>
        <li><a href="/">Home</a></li>
      </ul>
    </nav>
  </header>

  <section aria-labelledby="content-heading">
    <h2 id="content-heading">Main Content</h2>
    <!-- Content here -->
  </section>
</main>
```

### Accessible Forms
```html
<label for="email">Email Address</label>
<input
  type="email"
  id="email"
  name="email"
  required
  aria-describedby="email-help email-error">

<div id="email-help">We'll use this to create your account</div>
<div id="email-error" role="alert" aria-live="polite">
  <!-- Error messages appear here -->
</div>
```

### Focus Management
```css
/* Visible focus indicators */
button:focus,
input:focus,
select:focus,
textarea:focus {
  outline: 2px solid #007bff;
  outline-offset: 2px;
}
```

### ARIA for Custom Components
```html
<!-- Modal dialog -->
<div role="dialog" aria-labelledby="dialog-title" aria-modal="true">
  <h2 id="dialog-title">Confirm Action</h2>
  <button aria-label="Close dialog">×</button>
</div>
```

## Testing Requirements

### Automated Tools
- **axe-core**: JavaScript accessibility testing library
- **Lighthouse**: Chrome DevTools accessibility audit
- **WAVE**: Web accessibility evaluation tool

### Manual Testing Checklist
- [ ] Tab through all interactive elements
- [ ] Verify focus indicators are visible
- [ ] Test with screen reader (NVDA, JAWS, VoiceOver)
- [ ] Check color contrast ratios
- [ ] Verify form error messages are announced
- [ ] Test keyboard-only navigation

## Common Mistakes to Avoid

### ❌ Missing alt text
```html
<img src="chart.png"> <!-- No alt attribute -->
```

### ❌ Poor color contrast
```css
color: #999; /* Gray text on white background - too low contrast */
```

### ❌ Non-semantic markup
```html
<div class="button">Click me</div> <!-- Should be <button> -->
```

### ❌ Missing form labels
```html
<input type="text" placeholder="Name"> <!-- No associated label -->
```

## Legal Compliance

### Standards
- **WCAG 2.1 AA**: Recommended accessibility level
- **Section 508**: US government accessibility requirements
- **EN 301 549**: European accessibility standards

### Implementation Priority
1. **Semantic HTML** and proper heading structure
2. **Keyboard accessibility** and focus management
3. **Form accessibility** with proper labeling
4. **Color contrast** and visual indicators
5. **Screen reader compatibility**

By implementing these accessibility guidelines, Roo Code will create inclusive applications usable by everyone, including users with disabilities.