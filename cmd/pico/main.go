// Command pico is the pico code CLI agent.
package main

import "os"

func main() {
	if err := newRootCmd(os.Getenv).Execute(); err != nil {
		os.Exit(1)
	}
}
