# webrender
A very incomplete proof of concept GPU renderer for Servo

After updating shaders in webrender, go to servo and:

  1. Copy the webrender/res directory to servo/resources/shaders
  2. Go to the servo directory and do ./mach update-cargo -p webrender
  3. Create a pull request to servo and update the shaders in the servo repository.


# Use Webrender with Servo
To use a custom webrender with servo, go to your servo build directory and:

  1. Edit servo/components/servo/Cargo.toml
  2. Add at the end of the file:

```
paths = ["/Users/UserName/Path/To/webrender"]'
[replace]
"webrender:0.6.0" = { path = '/Users/UserName/Path/To/webrender/webrender/' }
"webrender_traits:0.6.0" = { path = '/Users/UserName/Path/To/webrender/webrender_traits' }
```

  3. Build as normal
