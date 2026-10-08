# GitHub About

Suggested description:

> An experimental, shape-checked ML programming language with a Rust compiler, tensor runtime, and automatic differentiation.

Apply using GitHub repository Settings/About, or with authorized GitHub API access:

```sh
gh repo edit rsasaki0109/ml-lang --description "An experimental, shape-checked ML programming language with a Rust compiler, tensor runtime, and automatic differentiation."
```

After environment publication, the GitHub API read succeeded, but applying
the description returned HTTP 403: `Resource not accessible by integration`.
The current integration cannot update repository metadata. Apply the prepared
description through the GitHub About editor, or use an integration authorized
for repository administration writes. No credentials are stored here.
