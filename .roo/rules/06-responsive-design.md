# Responsive Design Patterns

## Core Principles

### 1. Mobile-First Approach
- **Start small**: Design for mobile devices first, then enhance for larger screens
- **Progressive enhancement**: Add features as screen size increases
- **Content priority**: Focus on essential content and functionality

### 2. Flexible Layouts
- **Fluid grids**: Use relative units instead of fixed pixels
- **Flexible images**: Images that scale with their containers
- **Media queries**: Breakpoints for different screen sizes

### 3. Cross-Device Compatibility
- **Touch targets**: Adequate size for finger interaction (44px minimum)
- **Readable text**: Appropriate font sizes across devices
- **Navigation adaptation**: Different navigation patterns for different screens

## Implementation Patterns

### Fluid Typography
```css
/* Responsive font sizes */
html {
  font-size: 16px;
}

body {
  font-size: clamp(1rem, 2.5vw, 1.25rem);
  line-height: 1.6;
}

/* Heading scale */
h1 { font-size: clamp(2rem, 5vw, 3rem); }
h2 { font-size: clamp(1.5rem, 4vw, 2.25rem); }
h3 { font-size: clamp(1.25rem, 3vw, 1.75rem); }
```

### Flexible Grid Systems
```css
/* CSS Grid responsive layout */
.container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: clamp(1rem, 3vw, 2rem);
}

/* Flexbox responsive patterns */
.flex-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

@media (min-width: 768px) {
  .flex-container {
    flex-direction: row;
  }
}
```

### Responsive Images
```html
<!-- Responsive images -->
<img
  src="image.jpg"
  alt="Description"
  srcset="image-400.jpg 400w,
          image-800.jpg 800w,
          image-1200.jpg 1200w"
  sizes="(max-width: 768px) 100vw,
         (max-width: 1200px) 50vw,
         33vw">

<!-- Picture element for art direction -->
<picture>
  <source media="(min-width: 1200px)" srcset="hero-large.jpg">
  <source media="(min-width: 768px)" srcset="hero-medium.jpg">
  <img src="hero-small.jpg" alt="Hero image">
</picture>
```

## Breakpoint Strategy

### Common Breakpoints
```css
/* Mobile-first breakpoints */
@media (min-width: 576px) { /* Small tablets */ }
@media (min-width: 768px) { /* Tablets */ }
@media (min-width: 992px) { /* Small desktops */ }
@media (min-width: 1200px) { /* Large desktops */ }
@media (min-width: 1400px) { /* Extra large screens */ }
```

### Content-Based Breakpoints
```css
/* Break when content needs it */
@media (min-width: 50em) { /* When sidebar fits */ }
@media (min-width: 60em) { /* When 3-column layout works */ }
```

## Navigation Patterns

### Mobile Navigation
```html
<!-- Hamburger menu -->
<button
  class="mobile-menu-toggle"
  aria-expanded="false"
  aria-controls="main-nav">
  <span class="sr-only">Menu</span>
  <span class="hamburger"></span>
</button>

<nav id="main-nav" class="mobile-nav" hidden>
  <ul>
    <li><a href="/">Home</a></li>
    <li><a href="/products">Products</a></li>
  </ul>
</nav>
```

### Responsive Navigation
```css
/* Desktop navigation */
.main-nav {
  display: flex;
}

.main-nav ul {
  display: flex;
  gap: 2rem;
}

/* Mobile navigation */
@media (max-width: 767px) {
  .main-nav {
    position: fixed;
    top: 0;
    left: -100%;
    width: 80%;
    height: 100vh;
    background: white;
    transition: left 0.3s ease;
  }

  .main-nav.open {
    left: 0;
  }
}
```

## Touch-Friendly Design

### Touch Target Sizes
```css
/* Minimum touch targets */
button,
a,
input,
select,
textarea {
  min-height: 44px;
  min-width: 44px;
}

/* Comfortable touch targets */
.touch-target {
  min-height: 48px;
  min-width: 48px;
  padding: 0.75rem 1rem;
}
```

### Touch Gestures
```javascript
// Touch gesture handling
let touchStartX = 0;
let touchStartY = 0;

element.addEventListener('touchstart', (e) => {
  touchStartX = e.touches[0].clientX;
  touchStartY = e.touches[0].clientY;
});

element.addEventListener('touchend', (e) => {
  const touchEndX = e.changedTouches[0].clientX;
  const touchEndY = e.changedTouches[0].clientY;

  const deltaX = touchEndX - touchStartX;
  const deltaY = touchEndY - touchStartY;

  if (Math.abs(deltaX) > Math.abs(deltaY)) {
    // Horizontal swipe
    if (deltaX > 50) handleSwipeRight();
    if (deltaX < -50) handleSwipeLeft();
  }
});
```

## Performance Considerations

### Efficient Media Loading
```html
<!-- Lazy loading images -->
<img
  loading="lazy"
  src="image.jpg"
  alt="Description">

<!-- Responsive video -->
<video
  poster="video-poster.jpg"
  preload="none">
  <source src="video.mp4" type="video/mp4" media="(min-width: 768px)">
  <source src="video-mobile.mp4" type="video/mp4">
</video>
```

### Conditional Loading
```javascript
// Load components based on screen size
function loadComponents() {
  if (window.innerWidth >= 768) {
    import('./desktop-components.js');
  } else {
    import('./mobile-components.js');
  }
}

// Debounced resize handler
let resizeTimeout;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(loadComponents, 250);
});
```

## Testing Strategy

### Device Testing
- **Physical devices**: Test on actual mobile devices and tablets
- **Browser dev tools**: Use device emulation for quick testing
- **Cross-browser testing**: Ensure compatibility across browsers

### Responsive Testing Checklist
- [ ] **Viewport meta tag**: `<meta name="viewport" content="width=device-width, initial-scale=1">`
- [ ] **Fluid layouts**: Content adapts to different screen sizes
- [ ] **Readable text**: Font sizes appropriate for screen size
- [ ] **Touch targets**: Adequate size for finger interaction
- [ ] **Image scaling**: Images don't overflow containers
- [ ] **Navigation**: Works on all screen sizes
- [ ] **Performance**: Fast loading on mobile connections

### Automated Testing
```javascript
// Puppeteer responsive testing
async function testResponsiveDesign() {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Test different viewport sizes
  const viewports = [
    { width: 375, height: 667, deviceScaleFactor: 2 }, // iPhone
    { width: 768, height: 1024, deviceScaleFactor: 1 }, // iPad
    { width: 1920, height: 1080, deviceScaleFactor: 1 } // Desktop
  ];

  for (const viewport of viewports) {
    await page.setViewport(viewport);
    await page.goto('https://your-app.com');

    // Test layout and functionality
    const contentWidth = await page.$eval('.container',
      el => el.offsetWidth);
    expect(contentWidth).toBeLessThan(viewport.width);
  }

  await browser.close();
}
```

## Best Practices

### Design Principles
1. **Content first**: Design for content, not specific devices
2. **Progressive enhancement**: Start simple, add complexity
3. **Performance aware**: Optimize for mobile networks
4. **Inclusive design**: Consider all users and devices

### Technical Guidelines
- **Use relative units**: em, rem, %, vw, vh over px
- **Flexible images**: max-width: 100%, height: auto
- **Readable text**: Minimum 16px font size on mobile
- **Touch-friendly**: Adequate spacing and target sizes

### Maintenance
- **Regular testing**: Check layouts on new devices
- **Performance monitoring**: Track loading times across devices
- **User feedback**: Monitor usage patterns and issues
- **Continuous improvement**: Update breakpoints and layouts as needed

By following these responsive design patterns, Roo Code will create applications that work seamlessly across all devices and screen sizes.