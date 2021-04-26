// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: grpc.proto

package org.powertac.samplebroker.grpc;

/**
 * Protobuf type {@code XmlMessage}
 */
public  final class XmlMessage extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:XmlMessage)
    XmlMessageOrBuilder {
private static final long serialVersionUID = 0L;
  // Use XmlMessage.newBuilder() to construct.
  private XmlMessage(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private XmlMessage() {
    counter_ = 0;
    rawMessage_ = "";
    parsedInJava_ = false;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private XmlMessage(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    if (extensionRegistry == null) {
      throw new java.lang.NullPointerException();
    }
    int mutable_bitField0_ = 0;
    com.google.protobuf.UnknownFieldSet.Builder unknownFields =
        com.google.protobuf.UnknownFieldSet.newBuilder();
    try {
      boolean done = false;
      while (!done) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            done = true;
            break;
          default: {
            if (!parseUnknownFieldProto3(
                input, unknownFields, extensionRegistry, tag)) {
              done = true;
            }
            break;
          }
          case 8: {

            counter_ = input.readInt32();
            break;
          }
          case 18: {
            java.lang.String s = input.readStringRequireUtf8();

            rawMessage_ = s;
            break;
          }
          case 24: {

            parsedInJava_ = input.readBool();
            break;
          }
        }
      }
    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
      throw e.setUnfinishedMessage(this);
    } catch (java.io.IOException e) {
      throw new com.google.protobuf.InvalidProtocolBufferException(
          e).setUnfinishedMessage(this);
    } finally {
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.powertac.samplebroker.grpc.Grpc.internal_static_XmlMessage_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.powertac.samplebroker.grpc.Grpc.internal_static_XmlMessage_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.powertac.samplebroker.grpc.XmlMessage.class, org.powertac.samplebroker.grpc.XmlMessage.Builder.class);
  }

  public static final int COUNTER_FIELD_NUMBER = 1;
  private int counter_;
  /**
   * <code>int32 counter = 1;</code>
   */
  public int getCounter() {
    return counter_;
  }

  public static final int RAWMESSAGE_FIELD_NUMBER = 2;
  private volatile java.lang.Object rawMessage_;
  /**
   * <code>string rawMessage = 2;</code>
   */
  public java.lang.String getRawMessage() {
    java.lang.Object ref = rawMessage_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      rawMessage_ = s;
      return s;
    }
  }
  /**
   * <code>string rawMessage = 2;</code>
   */
  public com.google.protobuf.ByteString
      getRawMessageBytes() {
    java.lang.Object ref = rawMessage_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      rawMessage_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int PARSEDINJAVA_FIELD_NUMBER = 3;
  private boolean parsedInJava_;
  /**
   * <code>bool parsedInJava = 3;</code>
   */
  public boolean getParsedInJava() {
    return parsedInJava_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (counter_ != 0) {
      output.writeInt32(1, counter_);
    }
    if (!getRawMessageBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, rawMessage_);
    }
    if (parsedInJava_ != false) {
      output.writeBool(3, parsedInJava_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (counter_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(1, counter_);
    }
    if (!getRawMessageBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, rawMessage_);
    }
    if (parsedInJava_ != false) {
      size += com.google.protobuf.CodedOutputStream
        .computeBoolSize(3, parsedInJava_);
    }
    size += unknownFields.getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof org.powertac.samplebroker.grpc.XmlMessage)) {
      return super.equals(obj);
    }
    org.powertac.samplebroker.grpc.XmlMessage other = (org.powertac.samplebroker.grpc.XmlMessage) obj;

    boolean result = true;
    result = result && (getCounter()
        == other.getCounter());
    result = result && getRawMessage()
        .equals(other.getRawMessage());
    result = result && (getParsedInJava()
        == other.getParsedInJava());
    result = result && unknownFields.equals(other.unknownFields);
    return result;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    hash = (37 * hash) + COUNTER_FIELD_NUMBER;
    hash = (53 * hash) + getCounter();
    hash = (37 * hash) + RAWMESSAGE_FIELD_NUMBER;
    hash = (53 * hash) + getRawMessage().hashCode();
    hash = (37 * hash) + PARSEDINJAVA_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(
        getParsedInJava());
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.powertac.samplebroker.grpc.XmlMessage parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.powertac.samplebroker.grpc.XmlMessage parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(org.powertac.samplebroker.grpc.XmlMessage prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @java.lang.Override
  protected Builder newBuilderForType(
      com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * Protobuf type {@code XmlMessage}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:XmlMessage)
      org.powertac.samplebroker.grpc.XmlMessageOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.powertac.samplebroker.grpc.Grpc.internal_static_XmlMessage_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.powertac.samplebroker.grpc.Grpc.internal_static_XmlMessage_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.powertac.samplebroker.grpc.XmlMessage.class, org.powertac.samplebroker.grpc.XmlMessage.Builder.class);
    }

    // Construct using org.powertac.samplebroker.grpc.XmlMessage.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
      }
    }
    public Builder clear() {
      super.clear();
      counter_ = 0;

      rawMessage_ = "";

      parsedInJava_ = false;

      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.powertac.samplebroker.grpc.Grpc.internal_static_XmlMessage_descriptor;
    }

    public org.powertac.samplebroker.grpc.XmlMessage getDefaultInstanceForType() {
      return org.powertac.samplebroker.grpc.XmlMessage.getDefaultInstance();
    }

    public org.powertac.samplebroker.grpc.XmlMessage build() {
      org.powertac.samplebroker.grpc.XmlMessage result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.powertac.samplebroker.grpc.XmlMessage buildPartial() {
      org.powertac.samplebroker.grpc.XmlMessage result = new org.powertac.samplebroker.grpc.XmlMessage(this);
      result.counter_ = counter_;
      result.rawMessage_ = rawMessage_;
      result.parsedInJava_ = parsedInJava_;
      onBuilt();
      return result;
    }

    public Builder clone() {
      return (Builder) super.clone();
    }
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return (Builder) super.setField(field, value);
    }
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return (Builder) super.clearField(field);
    }
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return (Builder) super.clearOneof(oneof);
    }
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, java.lang.Object value) {
      return (Builder) super.setRepeatedField(field, index, value);
    }
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return (Builder) super.addRepeatedField(field, value);
    }
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof org.powertac.samplebroker.grpc.XmlMessage) {
        return mergeFrom((org.powertac.samplebroker.grpc.XmlMessage)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.powertac.samplebroker.grpc.XmlMessage other) {
      if (other == org.powertac.samplebroker.grpc.XmlMessage.getDefaultInstance()) return this;
      if (other.getCounter() != 0) {
        setCounter(other.getCounter());
      }
      if (!other.getRawMessage().isEmpty()) {
        rawMessage_ = other.rawMessage_;
        onChanged();
      }
      if (other.getParsedInJava() != false) {
        setParsedInJava(other.getParsedInJava());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.powertac.samplebroker.grpc.XmlMessage parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.powertac.samplebroker.grpc.XmlMessage) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }

    private int counter_ ;
    /**
     * <code>int32 counter = 1;</code>
     */
    public int getCounter() {
      return counter_;
    }
    /**
     * <code>int32 counter = 1;</code>
     */
    public Builder setCounter(int value) {
      
      counter_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 counter = 1;</code>
     */
    public Builder clearCounter() {
      
      counter_ = 0;
      onChanged();
      return this;
    }

    private java.lang.Object rawMessage_ = "";
    /**
     * <code>string rawMessage = 2;</code>
     */
    public java.lang.String getRawMessage() {
      java.lang.Object ref = rawMessage_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        rawMessage_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>string rawMessage = 2;</code>
     */
    public com.google.protobuf.ByteString
        getRawMessageBytes() {
      java.lang.Object ref = rawMessage_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        rawMessage_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string rawMessage = 2;</code>
     */
    public Builder setRawMessage(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      rawMessage_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>string rawMessage = 2;</code>
     */
    public Builder clearRawMessage() {
      
      rawMessage_ = getDefaultInstance().getRawMessage();
      onChanged();
      return this;
    }
    /**
     * <code>string rawMessage = 2;</code>
     */
    public Builder setRawMessageBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      rawMessage_ = value;
      onChanged();
      return this;
    }

    private boolean parsedInJava_ ;
    /**
     * <code>bool parsedInJava = 3;</code>
     */
    public boolean getParsedInJava() {
      return parsedInJava_;
    }
    /**
     * <code>bool parsedInJava = 3;</code>
     */
    public Builder setParsedInJava(boolean value) {
      
      parsedInJava_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>bool parsedInJava = 3;</code>
     */
    public Builder clearParsedInJava() {
      
      parsedInJava_ = false;
      onChanged();
      return this;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFieldsProto3(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:XmlMessage)
  }

  // @@protoc_insertion_point(class_scope:XmlMessage)
  private static final org.powertac.samplebroker.grpc.XmlMessage DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.powertac.samplebroker.grpc.XmlMessage();
  }

  public static org.powertac.samplebroker.grpc.XmlMessage getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<XmlMessage>
      PARSER = new com.google.protobuf.AbstractParser<XmlMessage>() {
    public XmlMessage parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new XmlMessage(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<XmlMessage> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<XmlMessage> getParserForType() {
    return PARSER;
  }

  public org.powertac.samplebroker.grpc.XmlMessage getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
