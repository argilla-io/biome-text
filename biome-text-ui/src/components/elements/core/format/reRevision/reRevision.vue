<template>
  <p v-if="revision">Current rev.{{revision.substr(0,7)}}</p>
</template>

<script>
export default {
  name: 're-revision',
  data: () => ({
  }),
  props: ['revision'],
};
</script>
