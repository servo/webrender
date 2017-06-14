# WebRender
GPU renderer for the Web content, used by Servo.

## Update as a Dependency
After updating shaders in WebRender, go to servo and:

  * Go to the servo directory and do ./mach update-cargo -p webrender
  * Create a pull request to servo


## Use WebRender with Servo
To use a custom WebRender with servo, go to your servo build directory and:

  * Edit Cargo.toml
  * Add at the end of the file:

```
[replace]
"https://github.com/servo/webrender#0.36.0" = { path = 'Path/To/webrender/webrender/' }
"https://github.com/servo/webrender#webrender_api:0.36.0" = { path = 'Path/To/webrender/webrender_api' }
```

The exact replace references can be obtained with `cargo pkgid webrender`/`cargo pkgid webrender_api` command.

  * Build as normal

## Documentation

The Wiki has a [few pages](https://github.com/servo/webrender/wiki/) describing the internals and conventions of WebRender.

