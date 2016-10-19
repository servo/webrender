# webrender
A very incomplete proof of concept GPU renderer for Servo

After updating shaders in webrender, go to servo and:

  * Copy the webrender/res directory to servo/resources/shaders
  * Go to the servo directory and do ./mach update-cargo -p webrender
  * Create a pull request to servo and update the shaders in the servo repository.


# Use Webrender with Servo
To use a custom webrender with servo, go to your servo build directory and:

  * Edit servo/components/servo/Cargo.toml
  * Add at the end of the file:

```
[replace]
"webrender:0.6.0" = { path = '/Users/UserName/Path/To/webrender/webrender/' }
"webrender_traits:0.6.0" = { path = '/Users/UserName/Path/To/webrender/webrender_traits' }
```

  * Build as normal
