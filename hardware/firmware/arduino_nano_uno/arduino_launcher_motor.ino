#include <Arduino.h>
#include <ctype.h>
#include <string.h>

const uint8_t IN1_PIN = 2;
const uint8_t PWM_PIN = 3;
const uint8_t IN2_PIN = 4;

const uint8_t MOTOR_PWM = 220;
const unsigned long MOVE_10_DEG_MS = 500;

const float ANGLE_STEP_DEG = 10.0f;
const float MIN_ANGLE_DEG = -47.0f;
const float MAX_ANGLE_DEG = 87.0f;

float currentAngleDeg = 0.0f;
float moveStartAngleDeg = 0.0f;
float targetAngleDeg = 0.0f;
unsigned long moveStartedAtMs = 0;
bool moving = false;
int8_t moveDirection = 0;

char inputBuffer[32];
uint8_t inputLength = 0;

void motorOff() {
  analogWrite(PWM_PIN, 0);
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);
}

void printStatus() {
  Serial.print(F("STATUS angle_deg="));
  Serial.print(currentAngleDeg, 1);
  Serial.print(F(" target_deg="));
  Serial.print(moving ? targetAngleDeg : currentAngleDeg, 1);
  Serial.print(F(" state="));
  if (!moving) {
    Serial.println(F("STOPPED"));
  } else if (moveDirection > 0) {
    Serial.println(F("RIGHT"));
  } else {
    Serial.println(F("LEFT"));
  }
}

void updatePartialAngle() {
  if (!moving) return;

  unsigned long elapsed = millis() - moveStartedAtMs;
  if (elapsed > MOVE_10_DEG_MS) elapsed = MOVE_10_DEG_MS;

  float ratio = (float)elapsed / (float)MOVE_10_DEG_MS;
  currentAngleDeg = moveStartAngleDeg +
                    (moveDirection * ANGLE_STEP_DEG * ratio);
}

void stopMovement(bool acknowledge) {
  if (moving) updatePartialAngle();
  motorOff();
  moving = false;
  moveDirection = 0;

  if (acknowledge) {
    Serial.println(F("ACK STOP"));
    printStatus();
  }
}

void startMovement(int8_t direction) {
  if (moving) {
    Serial.println(F("ERR BUSY"));
    return;
  }

  float requestedAngle = currentAngleDeg + direction * ANGLE_STEP_DEG;
  if (requestedAngle < MIN_ANGLE_DEG || requestedAngle > MAX_ANGLE_DEG) {
    Serial.print(F("ERR ANGLE_LIMIT current_deg="));
    Serial.print(currentAngleDeg, 1);
    Serial.print(F(" requested_deg="));
    Serial.print(requestedAngle, 1);
    Serial.println(F(" range=-47.0:87.0"));
    return;
  }

  moveStartAngleDeg = currentAngleDeg;
  targetAngleDeg = requestedAngle;
  moveDirection = direction;
  moveStartedAtMs = millis();
  moving = true;

  if (direction > 0) {
    digitalWrite(IN1_PIN, HIGH);
    digitalWrite(IN2_PIN, LOW);
    Serial.print(F("ACK RIGHT target_deg="));
  } else {
    digitalWrite(IN1_PIN, LOW);
    digitalWrite(IN2_PIN, HIGH);
    Serial.print(F("ACK LEFT target_deg="));
  }

  analogWrite(PWM_PIN, MOTOR_PWM);
  Serial.println(targetAngleDeg, 1);
}

void finishMovementIfNeeded() {
  if (!moving || millis() - moveStartedAtMs < MOVE_10_DEG_MS) return;

  motorOff();
  currentAngleDeg = targetAngleDeg;
  moving = false;
  moveDirection = 0;

  Serial.print(F("ACK DONE angle_deg="));
  Serial.println(currentAngleDeg, 1);
  printStatus();
}

void processCommand(char *command) {
  while (*command == ' ' || *command == '\t') command++;

  char *end = command + strlen(command);
  while (end > command && (end[-1] == ' ' || end[-1] == '\t')) *--end = '\0';
  for (char *p = command; *p; ++p) *p = (char)toupper((unsigned char)*p);

  if (strcmp(command, "0") == 0 || strcmp(command, "STOP") == 0) {
    stopMovement(true);
  } else if (strcmp(command, "1") == 0 || strcmp(command, "RIGHT") == 0) {
    startMovement(1);
  } else if (strcmp(command, "2") == 0 || strcmp(command, "LEFT") == 0) {
    startMovement(-1);
  } else if (strcmp(command, "STATUS") == 0 || strcmp(command, "?") == 0) {
    updatePartialAngle();
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

void setup() {
  pinMode(IN1_PIN, OUTPUT);
  pinMode(PWM_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);
  motorOff();

  Serial.begin(9600);
  delay(300);
  Serial.println(F("ACK READY device=ARDUINO_LAUNCHER"));
  printStatus();
}

void loop() {
  readSerialCommands();
  finishMovementIfNeeded();
}
