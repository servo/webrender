# webrender
A very incomplete proof of concept GPU renderer for Servo

After updating shaders in webrender, go to servo and:

  1. Copy the webrender/res directory to servo/resources/shaders
  2. Go to the servo directoy and do ./mach update-cargo -p webrender
  3. Create a pull request to servo and update the shaders in the servo repository.
