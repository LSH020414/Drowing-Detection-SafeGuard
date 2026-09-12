#include <Arduino.h>
#include <ctype.h>
#include <string.h>

const uint8_t PWM1_PIN = 25;
const uint8_t DIR1_PIN = 26;
const uint8_t PWM2_PIN = 32;
const uint8_t DIR2_PIN = 33;

const uint32_t PWM_FREQUENCY = 20000;
const uint8_t PWM_BITS = 8;
const uint8_t PWM1_CHANNEL = 0;
const uint8_t PWM2_CHANNEL = 1;

const int SPEED_STEP_PERCENT = 5;
const int MAX_SPEED_PERCENT = 80;
const int KICK_PWM = 255;
const unsigned long KICK_TIME_MS = 450;

int targetSpeedPercent = 0;
int appliedPwm = 0;
bool kickStarting = false;
unsigned long kickStartedAtMs = 0;

char inputBuffer[32];
uint8_t inputLength = 0;

int percentToPwm(int percent) {
  return (percent * 255 + 50) / 100;
}

void writePwm(int value) {
  value = constrain(value, 0, 255);
  appliedPwm = value;

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PWM1_PIN, value);
  ledcWrite(PWM2_PIN, value);
#else
  ledcWrite(PWM1_CHANNEL, value);
  ledcWrite(PWM2_CHANNEL, value);
#endif
}

void printStatus() {
  Serial.print(F("STATUS speed_percent="));
  Serial.print(targetSpeedPercent);
  Serial.print(F(" applied_pwm="));
  Serial.print(appliedPwm);
  Serial.print(F(" state="));

  if (kickStarting) {
    Serial.println(F("KICKSTART"));
  } else if (targetSpeedPercent == 0) {
    Serial.println(F("STOPPED"));
  } else {
    Serial.println(F("RUNNING"));
  }
}

void stopMotors(bool acknowledge) {
  targetSpeedPercent = 0;
  kickStarting = false;
  writePwm(0);

  if (acknowledge) {
    Serial.println(F("ACK STOP speed_percent=0"));
    printStatus();
  }
}

void speedUp() {
  if (targetSpeedPercent >= MAX_SPEED_PERCENT) {
    Serial.println(F("ERR SPEED_LIMIT current_percent=80 max_percent=80"));
    return;
  }

  bool startingFromStop = (targetSpeedPercent == 0);
  targetSpeedPercent += SPEED_STEP_PERCENT;
  if (targetSpeedPercent > MAX_SPEED_PERCENT) {
    targetSpeedPercent = MAX_SPEED_PERCENT;
  }

  Serial.print(F("ACK SPEED_UP target_percent="));
  Serial.println(targetSpeedPercent);

  if (startingFromStop) {
    kickStarting = true;
    kickStartedAtMs = millis();
    writePwm(KICK_PWM);
    printStatus();
  } else if (!kickStarting) {
    writePwm(percentToPwm(targetSpeedPercent));
    printStatus();
  }
}

void speedDown() {
  if (targetSpeedPercent <= 0) {
    Serial.println(F("ERR SPEED_LIMIT current_percent=0 min_percent=0"));
    return;
  }

  targetSpeedPercent -= SPEED_STEP_PERCENT;
  Serial.print(F("ACK SPEED_DOWN target_percent="));
  Serial.println(targetSpeedPercent);

  if (targetSpeedPercent == 0) {
    kickStarting = false;
    writePwm(0);
  } else if (!kickStarting) {
    writePwm(percentToPwm(targetSpeedPercent));
  }
  printStatus();
}

void finishKickStartIfNeeded() {
  if (!kickStarting || millis() - kickStartedAtMs < KICK_TIME_MS) return;

  kickStarting = false;
  writePwm(percentToPwm(targetSpeedPercent));
  Serial.print(F("ACK KICKSTART_DONE target_percent="));
  Serial.println(targetSpeedPercent);
  printStatus();
}

void processCommand(char *command) {
  while (*command == ' ' || *command == '\t') command++;

  char *end = command + strlen(command);
  while (end > command && (end[-1] == ' ' || end[-1] == '\t')) *--end = '\0';
  for (char *p = command; *p; ++p) *p = (char)toupper((unsigned char)*p);

  if (strcmp(command, "0") == 0 || strcmp(command, "STOP") == 0) {
    stopMotors(true);
  } else if (strcmp(command, "1") == 0 || strcmp(command, "SPEED_UP") == 0) {
    speedUp();
  } else if (strcmp(command, "2") == 0 || strcmp(command, "SPEED_DOWN") == 0) {
    speedDown();
  } else if (strcmp(command, "STATUS") == 0 || strcmp(command, "?") == 0) {
    printStatus();
  } else if (*command != '\0') {
    Serial.print(F("ERR UNKNOWN_COMMAND value="));
    Serial.println(command);
  }
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (inputLength > 0) {
        inputBuffer[inputLength] = '\0';
        processCommand(inputBuffer);
        inputLength = 0;
      }
    } else if (inputLength < sizeof(inputBuffer) - 1) {
      inputBuffer[inputLength++] = c;
    } else {
      inputLength = 0;
      Serial.println(F("ERR COMMAND_TOO_LONG"));
    }
  }
}

void setupPwm() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PWM1_PIN, PWM_FREQUENCY, PWM_BITS);
  ledcAttach(PWM2_PIN, PWM_FREQUENCY, PWM_BITS);
#else
  ledcSetup(PWM1_CHANNEL, PWM_FREQUENCY, PWM_BITS);
  ledcSetup(PWM2_CHANNEL, PWM_FREQUENCY, PWM_BITS);
  ledcAttachPin(PWM1_PIN, PWM1_CHANNEL);
  ledcAttachPin(PWM2_PIN, PWM2_CHANNEL);
#endif
}

void setup() {
  Serial.begin(115200);

  pinMode(DIR1_PIN, OUTPUT);
  pinMode(DIR2_PIN, OUTPUT);
  digitalWrite(DIR1_PIN, HIGH);
  digitalWrite(DIR2_PIN, LOW);

  setupPwm();
  stopMotors(false);

  delay(300);
  Serial.println(F("ACK READY device=ESP32_BLDC"));
  printStatus();
}

void loop() {
  readSerialCommands();
  finishKickStartIfNeeded();
}
