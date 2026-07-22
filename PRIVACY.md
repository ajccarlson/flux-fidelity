# Privacy

Last updated: 2026-07-21.

FSRCNNX-EXT processes readable video and, when enabled, image pixels from the current top-level page to render enhanced output locally. It also observes element geometry, playback timing, and the page origin needed to apply the user's settings. This data is used only for the extension's video and image enhancement features.

The developer does not collect or receive browsing activity, page content, media, identifiers, diagnostics, or usage analytics. The extension has no telemetry, advertising, account, or external service. It does not transmit page or media data to the developer or any third party, and it executes no remotely hosted code. Runtime models, shaders, and supporting files are bundled with the extension.

User choices are stored in `chrome.storage.local` by exact page origin. They remain on the device until the extension's local data is cleared or the extension is removed. Video frames, image pixels, playback measurements, tab identifiers, and document identifiers are not written to persistent storage.

## Permissions

- **Site access:** The content script runs on user-allowed HTTP, HTTPS, and local-file pages so it can find eligible media, restore that origin's settings, and maintain rendering across page lifecycle changes. It runs only in the top-level frame. Browser controls can restrict the sites on which it runs; local-file access requires a separate browser opt-in.
- **Active tab:** Opening the toolbar action temporarily allows the popup to identify and communicate with the current tab.
- **Storage:** Saves the user's settings separately for each page origin on the local device.
- **Bundled web-accessible resources:** Lets the isolated content script load only the extension's packaged runtime modules, shaders, models, and supporting files. It does not grant a remote service access to user data.

FSRCNNX-EXT does not request cookie, browsing-history, web-request, download, clipboard, notification, or background network-access permissions. Websites and the browser may perform their own network activity independently of the extension.
