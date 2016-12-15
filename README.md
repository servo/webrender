# webrender
GPU renderer for the Web content, used by Servo.

## Update as a Dependency
After updating shaders in webrender, go to servo and:

  * Go to the servo directory and do ./mach update-cargo -p webrender
  * Create a pull request to servo


## Use Webrender with Servo
To use a custom webrender with servo, go to your servo build directory and:

  * Edit Cargo.toml
  * Add at the end of the file:

```
[replace]
"https://github.com/servo/webrender#0.11.0" = { path = 'Path/To/webrender/webrender/' }
"https://github.com/servo/webrender#webrender_traits:0.11.0" = { path = 'Path/To/webrender/webrender_traits' }
```

  * Build as normal

## Documentation

[Coordinate Systems](https://github.com/servo/webrender/wiki/Coordinate-Systems)
