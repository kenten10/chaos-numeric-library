# Security Policy

## Supported versions

Until the first stable release, security fixes are applied to the latest release only.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's **Security** tab using a private
vulnerability report. Do not open a public issue for an unpatched vulnerability.

Include the affected version, reproduction steps, impact, and any suggested mitigation. We aim
to acknowledge reports within seven days and will coordinate disclosure after a fix is available.

## Persistence trust boundary

`load_result` validates metadata size, NPZ structure, array sizes, descriptors, and SHA-256
integrity before constructing result objects. These checks limit accidental or malicious resource
use, but persisted experiment directories are not a sandbox. Do not share an output directory with
untrusted writers, and do not load artifacts while another process is modifying them.
