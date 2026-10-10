// ESP32 node firmware (verified in Wokwi): DHT22 + PIR + LDR -> JSON over MQTT
#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>

#define DHT_PIN 15
#define PIR_PIN 13
#define LDR_PIN 34
#define NODE_ID 1

const char* broker = "broker.hivemq.com";               // public test broker, demo only
const char* topic  = "wsn_aura_cartographer/node1/data"; // keep unique

WiFiClient wifi;
PubSubClient mqtt(wifi);
DHT dht(DHT_PIN, DHT22);

void connectAll() {
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.begin("Wokwi-GUEST", "", 6);
    while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
    Serial.println("\nWiFi connected");
  }
  mqtt.setServer(broker, 1883);
  while (!mqtt.connected()) {
    if (mqtt.connect("wsn_aura_node1_x7")) Serial.println("MQTT connected");
    else { Serial.println("MQTT retry"); delay(2000); }
  }
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  pinMode(PIR_PIN, INPUT);
  connectAll();
}

void loop() {
  connectAll();
  mqtt.loop();
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  int light = map(analogRead(LDR_PIN), 0, 4095, 0, 450);
  int motion = digitalRead(PIR_PIN);
  String p = "{\"node_id\":" + String(NODE_ID) +
             ",\"temperature\":" + (isnan(t) ? "null" : String(t, 1)) +
             ",\"humidity\":" + (isnan(h) ? "null" : String(h, 1)) +
             ",\"light\":" + String(light) +
             ",\"motion\":" + String(motion) + "}";
  mqtt.publish(topic, p.c_str());
  Serial.println(p);
  delay(2000);
}
