// default component to create the buttons to click on in the mental arithmetic task which are arranged in a circle

import React from 'react';

export default function MentalArithmeticAnswers(props) {

    const styles = {
        transform: `rotate(${props.angle}deg) translate(13em) rotate(-${props.angle}deg)`,
        // display: "block",
        // overflow: "hidden",
        //position: "absolute",
        //top: "50%",
        //left: "50%",
        width: "4em",
        height: "4em",
        margin: "-2em",
        /* half the width */
        border: "solid thin"
    };


    return (<div className="button is-dark is-outlined" style={styles} onClick={props.onClick}>{props.num}</div>);
}