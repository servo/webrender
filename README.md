# webrender
A very incomplete proof of concept GPU renderer for Servo

After updating shaders in webrender, go to servo and:

  1. Copy the webrender/res directory to servo/resources/shaders
  2. Go to the servo directoy and do ./mach update-cargo -p webrender
  3. Create a pull request to servo and update the shaders in the servo repository.


# Use Webrender with Servo
To use a custom webrender with servo, go to your servo build directory and:

  1. Edit servo/.cargo/config - Create this file if it doesn't exist already.
  2. Add 'paths = ["/Users/UserName/Path/To/webrender"]'
  3. Build as normal
